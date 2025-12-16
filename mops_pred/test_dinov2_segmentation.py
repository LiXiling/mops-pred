import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from mops_pred.datasets.dataset_factory import create_dataloader
from mops_pred.models.dinov2_segmentation import DINOv2SegmentationModel

# Configuration for zero-shot testing
MODEL_NAME = "dinov2_vitb14"  # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
BATCH_SIZE = 16
NUM_CLASSES = 56  # Semantic segmentation classes for clutter dataset
TASK = "affordance"  # or "affordance"
MULTILABEL = True  # True for affordance, False for semantic


def visualize_segmentation(images, predictions, targets, num_samples=4):
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
    plt.savefig(
        "dinov2_zeroshot_segmentation_results.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
    print("Visualization saved to dinov2_zeroshot_segmentation_results.png")


def test_dinov2_segmentation_zeroshot():
    """
    Tests DINOv2 segmentation in zero-shot mode (frozen backbone + linear head).
    """
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(42)

    # Initialize model with frozen backbone (zero-shot mode)
    model = DINOv2SegmentationModel(
        num_classes=NUM_CLASSES,
        task=TASK,
        model_name=MODEL_NAME,
        freeze_backbone=True,  # Freeze for zero-shot
        lr=1e-3,  # Higher LR for training just the head
        multilabel=MULTILABEL,
    )

    # Create dataloaders
    train_dl, test_dl = create_dataloader(
        {
            "dataset": {
                "name": "clutter",
                "alias": "kitchen_affordance",
                "data_dir": "data/mops_data/mops_kitchen_dataset_v3_train.h5",
                "test_dir": "data/mops_data/mops_kitchen_dataset_v3_test_new.h5",
                "labels": [TASK],
            }
        },
        batch_size=BATCH_SIZE,
        augment=False,
    )

    print(f"\n{'=' * 60}")
    print("DINOv2 Zero-Shot Segmentation Testing")
    print(f"Model: {MODEL_NAME}")
    print(f"Task: {TASK}")
    print(f"Num Classes: {NUM_CLASSES}")
    print("Backbone: FROZEN (zero-shot mode)")
    print(f"{'=' * 60}\n")

    # Quick training of just the segmentation head (linear probing)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/iou",
        dirpath="checkpoints",
        filename=f"dinov2-{MODEL_NAME}-zeroshot-{TASK}-best",
        save_top_k=1,
        mode="max",
    )

    trainer = L.Trainer(
        max_epochs=20,  # Quick training for zero-shot
        logger=True,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    print("Training segmentation head (linear probing)...")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    print("Linear probing complete.")

    # Load best model and validate
    print("\nLoading best model and running final validation...")
    best_model = DINOv2SegmentationModel.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    trainer.validate(best_model, dataloaders=test_dl)
    print("Final validation complete.")

    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    best_model.eval()
    for batch in test_dl:
        batch = {k: v.to(best_model.device) for k, v in batch.items()}
        with torch.no_grad():
            predictions = best_model.predict_step(batch, 0)["predictions"]

        visualize_segmentation(
            batch["image"],
            predictions,
            batch[TASK],
            num_samples=min(4, len(batch["image"])),
        )
        break

    print("\nZero-shot testing complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    test_dinov2_segmentation_zeroshot()
