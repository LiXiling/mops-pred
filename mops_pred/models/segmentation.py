import lightning as L
import torch
import torchmetrics
import torchmetrics.classification
from torch import nn, optim
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from .model_factory import register_model


@register_model(name="segmentation")
class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        task: str = "semantic",
        lr: float = 1e-4,
    ) -> None:
        """
        Initializes a DeepLabV3 model for semantic segmentation.

        Args:
            num_classes: Number of segmentation classes.
            task: The key for the target mask in the batch (e.g., 'semantic').
            lr: Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

        # Replace the classifier head with a new one for the correct number of classes
        in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        # Define metrics
        self.miou = torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=num_classes, average="macro"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = (
            batch[self.hparams.task].long().squeeze(1)
        )  # Target mask, remove channel dim

        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/iou", self.miou(logits, y), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = (
            batch[self.hparams.task].long().squeeze(1)
        )  # Target mask, remove channel dim

        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)

        self.log("val/loss", loss, on_epoch=True)
        self.log("val/iou", self.miou(logits, y), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
