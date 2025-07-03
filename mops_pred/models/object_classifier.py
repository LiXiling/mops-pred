import lightning as L
import torchmetrics
from torch import nn, optim

from .backbones.backbone_factory import BackboneABC
from .model_factory import register_model


@register_model(name="object_clf")
class ObjectClassifierModel(L.LightningModule):
    def __init__(
        self,
        backbone: BackboneABC,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.backbone = backbone
        clf_channels = backbone.out_channels
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(clf_channels, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.clf(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["class_label"]

        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train/loss", loss, on_step=True)
        self.log(
            "train/acc",
            torchmetrics.functional.accuracy(
                logits, y, task="multiclass", num_classes=self.num_classes
            ),
            on_epoch=True,
            batch_size=x.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["class_label"]
        pred = self.forward(x)
        acc = torchmetrics.functional.accuracy(
            pred, y, task="multiclass", num_classes=self.num_classes
        )
        self.log("val/acc", acc, on_epoch=True, batch_size=x.shape[0])

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
