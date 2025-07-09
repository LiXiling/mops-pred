from typing import List

import h5py
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as T_v2
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask

from mops_pred.datasets.dataset_factory import register_dataset


@register_dataset(name="clutter")
class ClutterDataset(Dataset):
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

        # Define transforms
        if self.augment:
            self.spatial_transform = T_v2.Compose(
                [
                    T_v2.RandomResizedCrop(
                        size=(224, 224),
                        scale=(0.8, 1.0),
                        ratio=(0.75, 1.33),
                        antialias=True,
                    ),
                    T_v2.RandomHorizontalFlip(p=0.5),
                    T_v2.RandomRotation(degrees=10),
                ]
            )
            self.color_transform = T_v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            )
        else:
            self.spatial_transform = T_v2.Resize(size=(224, 224), antialias=True)

        self.normalize_transform = T_v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Open file and get basic info
        with h5py.File(h5_path, "r") as f:
            # Get splits and filter indices
            splits = f["metadata"]["splits"][:]
            self.indices = np.where(splits == self.is_train)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        actual_idx = self.indices[idx]
        image_id = f"image_{actual_idx:06d}"

        # Load image and convert to tensor
        image_np = self.h5_file["images"][image_id][:]  # HWC, uint8
        image = F.to_tensor(image_np)  # CHW, float

        # Prepare return dictionary
        sample = {"image_id": image_id}
        masks = {}

        # Load requested labels
        for label_type in self.labels:
            print(f"Loading label type: {label_type}")
            if label_type == "class":
                class_idx = self.h5_file["labels"]["class_labels"][actual_idx]
                sample["class_label"] = torch.tensor(class_idx, dtype=torch.long)
                # sample["class_name"] = self.class_names[class_idx] # class_names not defined
            else:
                # Load mask and convert to tensor
                mask_data = self.h5_file["masks"][label_type][image_id][:]
                m = None
                if mask_data.ndim == 2:
                    m = torch.from_numpy(mask_data).unsqueeze(0)
                # Check if class channel is first or last
                elif mask_data.ndim == 3 and mask_data.shape[0] == mask_data.shape[1]:
                    # Channel-first format (C, H, W)
                    m = torch.from_numpy(mask_data).permute(2, 0, 1)
                else:
                    m = mask_data
                masks[label_type] = m

        # Apply spatial augmentations to image and all masks
        # We wrap masks to prevent them from being normalized or antialiased
        wrapped_masks = {k: Mask(v) for k, v in masks.items()}
        image, wrapped_masks = self.spatial_transform(image, wrapped_masks)
        print(wrapped_masks["semantic"].shape)

        # Apply color augmentations (only on image)
        if self.augment:
            image = self.color_transform(image)

        # Normalize image
        sample["image"] = self.normalize_transform(image)

        # Unwrap masks and add to sample
        for k, v in wrapped_masks.items():
            # Squeeze to remove channel dim for masks
            sample[k] = v

        return sample


if __name__ == "__main__":
    dataset_semantic = ClutterDataset(
        h5_path="data/mops_data/mops_clutter_dataset_v2.h5",
        train=True,
        labels=["semantic", "affordance"],
        augment=True,
    )

    # Test loading a sample
    sample = dataset_semantic[0]
    print(f"Sample keys: {sample.keys()}")
    for k, v in sample.items():
        print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
