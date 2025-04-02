
from transformers import ASTForAudioClassification, ConvNextForImageClassification
from torchmetrics import AUROC, MetricCollection
from birdset.modules.metrics.multilabel import TopKAccuracy, cmAP
import lightning as l
import torch.nn as nn
import torch
from transformers import AdamW


class ConvNextClassifierLightningModule(l.LightningModule):
    def __init__(
        self,
        num_classes,
        num_epochs,
    ):
        super(ConvNextClassifierLightningModule, self).__init__()
        self.model = ConvNextForImageClassification.from_pretrained(
            "DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.loss = nn.BCEWithLogitsLoss()
        self.main_metric = cmAP(num_labels=num_classes, thresholds=None)
        self.other_metrics = MetricCollection(
            {
                "MultilabelAUROC": AUROC(
                    task="multilabel",
                    num_labels=num_classes,
                    average="macro",
                    thresholds=None,
                ),
                "T1Accuracy": TopKAccuracy(topk=1),
            }
        )

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        values = batch["input_values"]
        labels = batch["labels"]
        logits = self(values)

        loss = self.loss(logits, labels)
        predictions = torch.sigmoid(logits)

        return loss, predictions

    def training_step(self, batch, batch_idx):
        loss, preds = self.common_step(batch, batch_idx)
        self.log(
            f"train/{self.loss.__class__.__name__}",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds = self.common_step(batch, batch_idx)
        self.log(
            f"val/{self.loss.__class__.__name__}",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        loss, preds = self.common_step(batch, batch_idx)
        self.log(
            f"test/{self.loss.__class__.__name__}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.main_metric(preds, batch["labels"].int())
        self.log(
            f"test/{self.main_metric.__class__.__name__}",
            self.main_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.other_metrics(preds, batch["labels"].int())
        self.log_dict(self.other_metrics, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)