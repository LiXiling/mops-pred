from typing import List

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from .dataset_factory import register_dataset


@register_dataset(name="object_centric")
class ObjectCentricDataset(Dataset):
    """PyTorch Dataset for HDF5 synthetic object dataset."""

    def __init__(
        self,
        h5_path: str,
        train: bool = True,
        labels: List[str] | None = None,
        augment: bool = False,
    ):
        """
        Args:
            h5_path: Path to HDF5 file
            train: True if Train split should be used, False for test split.
            labels: List of label types to return. Default None = ["class"] Options:
                - "class": class labels for classification
                - "semantic": semantic segmentation masks (object vs background)
                - "parts": part segmentation masks (different object parts)
                - "instance": instance segmentation masks (individual object instances)
                - "affordance": affordance segmentation masks (functional regions)
                - "depth": depth maps (distance from camera)
                - "normal": normal maps (surface normals)

            augment: If Image augmentations should be applied.
        """
        self.h5_path = h5_path
        self.is_train = train
        self.labels = labels if labels is not None else ["class"]
        self.augment = augment
        self.h5_file = None

        # Hardcoded transforms for reproducibility
        if self.augment:
            self.image_transform = T.Compose(
                [
                    T.ToPILImage(),
                    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    # Gaussien blur is not required with perfect rendering
                    # T.GaussianBlur(kernel_size=3),
                    T.RandomHorizontalFlip(0.5),
                    T.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                    ),
                    T.RandomRotation(10),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.image_transform = T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        # Open file and get basic info
        with h5py.File(h5_path, "r") as f:
            # Get class names and create mapping
            self.class_names = [
                name.decode("utf-8") for name in f["labels"]["class_names"][:]
            ]
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

            # Get splits and filter indices
            splits = f["labels"]["splits"][:]
            self.indices = np.where(splits == self.is_train)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        actual_idx = self.indices[idx]
        image_id = f"image_{actual_idx:06d}"

        # Load image
        image = self.h5_file["images"][image_id][:]

        # Prepare return dictionary
        sample = {"image": image, "image_id": image_id}

        # Load requested labels
        for label_type in self.labels:
            if label_type == "class":
                class_idx = self.h5_file["labels"]["class_labels"][actual_idx]
                sample["class_label"] = torch.tensor(class_idx, dtype=torch.long)
                sample["class_name"] = self.class_names[class_idx]
            else:
                sample[label_type] = self.h5_file["masks"][label_type][image_id][:]

        # Apply transforms to image
        sample["image"] = self.image_transform(sample["image"])

        return sample


if __name__ == "__main__":
    dataset_classification = ObjectCentricDataset(
        h5_path="data/mops_data/mops_single_dataset_v2.h5",
        train=True,
        labels=["class"],
        augment=True,
    )

    # Test loading a sample
    sample = dataset_classification[0]
    # print(f"Sample keys: {sample.keys()}")
    # print(f"Image shape: {sample["image"].shape}")
    # print(f"Class: {sample["class_name"]} (label: {sample["class_label"]})")
