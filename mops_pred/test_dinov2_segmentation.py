import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from mops_pred.config import DatasetConfig
from mops_pred.datasets.dataset_factory import create_dataloader
from mops_pred.models.dinov2_segmentation import DINOv2SegmentationModel
from mops_pred.models.dinov3_segmentation import DINOv3SegmentationModel

# Shared configuration for zero-shot testing
BATCH_SIZE = 16
NUM_CLASSES = 56  # Semantic segmentation classes for clutter dataset
TASK = "affordance"
MULTILABEL = True  # True for affordance, False for semantic

# Model identifiers
DINOV2_MODEL_NAME = "dinov2_vitb14"
DINOV3_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def visualize_segmentation(
    images, predictions, targets, num_samples: int = 4, output_path: str | None = None
):
    """Visualize segmentation predictions."""
    num_samples = min(num_samples, len(images))
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        axs = axs[np.newaxis, :]

    for i in range(num_samples):
        # Unnormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = images[i].cpu() * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()

        # Get prediction and target masks
        pred_mask = predictions[i].cpu().numpy()
        target_mask = targets[i].cpu().squeeze().numpy()

        # Plot
        axs[i, 0].imshow(img)
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(target_mask, cmap="tab20")
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(pred_mask, cmap="tab20")
        axs[i, 2].set_title("Prediction (Zero-Shot)")
        axs[i, 2].axis("off")

    plt.tight_layout()
    outfile = output_path or "dinov2_zeroshot_segmentation_results.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Visualization saved to {outfile}")


def _get_dataloaders():
    """Build train/test dataloaders for the kitchen affordance dataset."""
    return create_dataloader(
        DatasetConfig(
            name="clutter",
            alias="kitchen_affordance",
            data_dir="data/mops_data/mops_clutter_dataset_v2.h5",
            labels=[TASK],
        ),
        batch_size=BATCH_SIZE,
        augment=False,
    )


def _run_zero_shot(model_cls, model_kwargs, experiment_tag: str):
    """Shared zero-shot workflow for DINO variants."""
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(42)

    tag_for_files = experiment_tag.replace("/", "-")

    # Initialize model and data
    model = model_cls(**model_kwargs)
    train_dl, test_dl = _get_dataloaders()

    print(f"\n{'=' * 60}")
    print(f"Zero-Shot Segmentation Testing :: {tag_for_files}")
    print(f"Task: {TASK}")
    print(f"Num Classes: {NUM_CLASSES}")
    print("Backbone: FROZEN (linear probing)")
    print(f"{'=' * 60}\n")

    checkpoint_callback = ModelCheckpoint(
        monitor="val/iou",
        dirpath="checkpoints",
        filename=f"{tag_for_files}-best",
        save_top_k=1,
        mode="max",
    )

    trainer = L.Trainer(
        max_epochs=20,
        logger=True,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    print("Training segmentation head (linear probing)...")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    print("Linear probing complete.")

    # Load best checkpoint
    print("\nLoading best model and running final validation...")
    best_path = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    if best_path is None:
        raise RuntimeError("No checkpoint was saved during training.")

    best_model = model_cls.load_from_checkpoint(best_path)
    trainer.validate(best_model, dataloaders=test_dl)
    print("Final validation complete.")

    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    best_model.eval()
    device = getattr(trainer.strategy, "root_device", best_model.device)
    best_model.to(device)
    for batch in test_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            predictions = best_model.predict_step(batch, 0)["predictions"]

        visualize_segmentation(
            batch["image"],
            predictions,
            batch[TASK],
            num_samples=min(4, len(batch["image"])),
            output_path=f"{tag_for_files}_results.png",
        )
        break

    print("\nZero-shot testing complete!")
    print(f"Best model saved to: {best_path}")


def test_dinov2_segmentation_zeroshot():
    """DINOv2 frozen backbone + linear segmentation head."""
    _run_zero_shot(
        DINOv2SegmentationModel,
        {
            "num_classes": NUM_CLASSES,
            "task": TASK,
            "model_name": DINOV2_MODEL_NAME,
            "freeze_backbone": True,
            "lr": 1e-3,
            "multilabel": MULTILABEL,
        },
        experiment_tag=f"dinov2-{DINOV2_MODEL_NAME}-zeroshot-{TASK}",
    )


def test_dinov3_segmentation_zeroshot():
    """DINOv3 frozen backbone + linear segmentation head."""
    _run_zero_shot(
        DINOv3SegmentationModel,
        {
            "num_classes": NUM_CLASSES,
            "task": TASK,
            "model_name": DINOV3_MODEL_NAME,
            "freeze_backbone": True,
            "lr": 1e-3,
            "multilabel": MULTILABEL,
        },
        experiment_tag=f"dinov3-{DINOV3_MODEL_NAME}-zeroshot-{TASK}",
    )


if __name__ == "__main__":
    test_dinov3_segmentation_zeroshot()
