import pandas as pd
import random
from tqdm import tqdm
from collections import Counter
from dataclasses import dataclass
import torch
from torchaudio.transforms import Spectrogram, MelScale
from birdset.datamodule.components.resize import Resizer
from birdset.datamodule.components.augmentations import PowerToDB
import numpy as np
import torch_audiomentations
import torchvision.transforms
import lightning as L
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os


def smart_sampling(dataset, label_name, class_limit, event_limit):
    """
    Performs class-balanced sampling on a Hugging Face Dataset by limiting the number of samples 
    per class and ensuring that events within each file are not overrepresented.

    Args:
        dataset (datasets.Dataset): The input Hugging Face dataset containing audio/image/text samples.
        label_name (str): The name of the column containing class labels.
        class_limit (int or None): The maximum number of samples allowed per class. If None, no limit is applied.
        event_limit (int): The maximum occurrences of the same event within a file.

    Returns:
        datasets.Dataset: A subsampled dataset where:
            - No class exceeds `class_limit` samples.
            - No event within a file exceeds `event_limit` occurrences.

    Sampling Strategy:
    - Assigns a unique identifier ("id") to each sample based on its `filepath` and label.
    - Computes the count of occurrences for each class and event.
    - Iteratively reduces overrepresented events to match the given constraints.
    - Randomly selects a subset of samples based on the computed limits.
    - Returns a filtered dataset with balanced class distributions.

    Example Usage:
    ```
    from datasets import load_dataset

    dataset = load_dataset("custom_dataset")["train"]
    sampled_dataset = smart_sampling(dataset, label_name="class", class_limit=1000, event_limit=10)
    ```
    """
    def _unique_identifier(x, labelname):
        file = x["filepath"]
        label = x[labelname]
        return {"id": f"{file}-{label}"}

    class_limit = class_limit if class_limit else -float("inf")
    dataset = dataset.map(
        lambda x: _unique_identifier(x, label_name), desc="sampling: unique-identifier"
    )
    df = pd.DataFrame(dataset)
    path_label_count = df.groupby(["id", label_name], as_index=False).size()
    path_label_count = path_label_count.set_index("id")
    class_sizes = df.groupby(label_name).size()

    for label in tqdm(class_sizes.index, desc="sampling"):
        current = path_label_count[path_label_count[label_name] == label]
        total = current["size"].sum()
        most = current["size"].max()

        while total > class_limit or most != event_limit:
            largest_count = current["size"].value_counts()[current["size"].max()]
            n_largest = current.nlargest(largest_count + 1, "size")
            to_del = n_largest["size"].max() - n_largest["size"].min()

            idxs = n_largest[n_largest["size"] == n_largest["size"].max()].index
            if (
                total - (to_del * largest_count) < class_limit
                or most == event_limit
                or most == 1
            ):
                break
            for idx in idxs:
                current.at[idx, "size"] = current.at[idx, "size"] - to_del
                path_label_count.at[idx, "size"] = (
                    path_label_count.at[idx, "size"] - to_del
                )

            total = current["size"].sum()
            most = current["size"].max()

    event_counts = Counter(dataset["id"])

    all_file_indices = {label: [] for label in event_counts.keys()}
    for idx, label in enumerate(dataset["id"]):
        all_file_indices[label].append(idx)

    limited_indices = []
    for file, indices in all_file_indices.items():
        limit = path_label_count.loc[file]["size"]
        limited_indices.extend(random.sample(indices, limit))

    dataset = dataset.remove_columns("id")
    return dataset.select(limited_indices)


def classes_one_hot(batch, num_classes):
    """
    Converts class labels to one-hot encoding.

    This method takes a batch of data and converts the class labels to one-hot encoding.
    The one-hot encoding is a binary matrix representation of the class labels.

    Args:
        batch (dict): A batch of data. The batch should be a dictionary where the keys are the field names and the values are the field data.

    Returns:
        dict: The batch with the "labels" field converted to one-hot encoding. The keys are the field names and the values are the field data.
    """
    label_list = [y for y in batch["labels"]]
    class_one_hot_matrix = torch.zeros(
        (len(label_list), num_classes), dtype=torch.float
    )

    for class_idx, idx in enumerate(label_list):
        class_one_hot_matrix[class_idx, idx] = 1

    class_one_hot_matrix = torch.tensor(class_one_hot_matrix, dtype=torch.float32)
    return {"labels": class_one_hot_matrix}

@dataclass
class CustomProcessingConfig:
    spectrogram_conversion: Spectrogram | None = Spectrogram(
        n_fft=1024,
        hop_length=320,
        power=2.0,
    )
    resizer: Resizer | None = (
        Resizer(
            db_scale=True,
        ),
    )
    melscale_conversion: MelScale | None = (
        MelScale(
            n_mels=128,
            sample_rate=32000,
            n_stft=513,  # n_fft//2+1
        ),
    )
    dbscale_conversion: PowerToDB | None = (PowerToDB(),)
    normalize_spectrogram: bool = (True,)
    mean: float = (-4.268,)
    std: float = (-4.569,)


class BasicTransformsWrapper:
    def __init__(
        self,
        wav_transforms,
        spec_transforms,
        decoding,
        feature_extractor,
        nocall_sampler,
        processing_config=CustomProcessingConfig,
    ):
        self.wav_transforms = torch_audiomentations.Compose(
            transforms=wav_transforms, output_type="object_dict"
        )
        self.spec_transforms = torchvision.transforms.Compose(
            transforms=spec_transforms
        )
        self.processing = processing_config
        self.nocall_sampler = nocall_sampler
        self.decoding = decoding
        self.feature_extractor = feature_extractor

    def __call__(self, batch, **kwds):

        batch = self.decoding(batch)

        waveform_batch = self._get_waveform_batch(batch)

        input_values = waveform_batch["input_values"]
        input_values = input_values.unsqueeze(1)
        labels = torch.tensor(batch["labels"])

        input_values, labels = self._waveform_augmentation(input_values, labels)

        if self.nocall_sampler:
            input_values, labels = self.nocall_sampler(input_values, labels)

        if self.processing.spectrogram_conversion is not None:
            spectrograms = self.processing.spectrogram_conversion(input_values)

            if self.spec_transforms:
                spectrograms = self.spec_transforms(spectrograms)

            if self.processing.melscale_conversion:
                spectrograms = self.processing.melscale_conversion(spectrograms)

            if self.processing.dbscale_conversion:
                spectrograms = self.processing.dbscale_conversion(spectrograms)

            if self.processing.resizer:
                spectrograms = self.processing.resizer.resize_spectrogram_batch(
                    spectrograms
                )

            if self.processing.normalize_spectrogram:
                spectrograms = (
                    spectrograms - self.processing.mean
                ) / self.processing.std

            input_values = spectrograms

        # values in labels need to be of type float for further use
        labels = labels.to(torch.float16)

        return {"input_values": input_values, "labels": labels}

    def _get_waveform_batch(self, batch):
        waveform_batch = [audio["array"] for audio in batch["audio"]]

        # extract/pad/truncate
        max_length = int(
            int(self.feature_extractor.sampling_rate) * int(self.decoding.max_len)
        )
        waveform_batch = self.feature_extractor(
            waveform_batch,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return waveform_batch

    def _waveform_augmentation(self, input_values, labels):
        labels = labels.unsqueeze(1).unsqueeze(1)
        output_dict = self.wav_transforms(
            samples=input_values,
            sample_rate=self.feature_extractor.sampling_rate,
            targets=labels,
        )
        labels = output_dict.targets.squeeze(1).squeeze(1)

        return output_dict.samples, labels
    



class CustomDatamodule(L.LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, num_classes, task):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.task = task
        self.len_trainset = len(dataset["train"])

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset["valid"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    

def save_dataloader_tensors_as_png(dataloader: DataLoader, output_dir: str = "tensor_plots"):
    """
    Saves all tensors from a PyTorch DataLoader as PNG images.

    Args:
        dataloader (DataLoader): A PyTorch DataLoader instance.
        output_dir (str): Directory where images will be saved (default: "tensor_plots").

    The function iterates over all batches and all tensors in each batch, saving them as images.
    """
    os.makedirs(output_dir, exist_ok=True)

    img_count = 0  # Counter for saved images

    for batch_idx, batch in enumerate(dataloader):
        tensors = batch["input_values"]  # Extract all tensors in the batch

        for sample_idx, tensor in enumerate(tensors):
            tensor = tensor.squeeze().numpy()  # Convert to NumPy array

            # Plot the tensor
            plt.imshow(np.flipud(tensor))
            plt.axis("off")  # Hide axes for a clean image

            # Save as PNG
            filename = os.path.join(output_dir, f"tensor_{img_count:05d}.png")
            plt.savefig(filename, bbox_inches="tight", pad_inches=0)
            plt.close()

            img_count += 1  # Increment image count

    print(f"Saved {img_count} images in {output_dir}")
