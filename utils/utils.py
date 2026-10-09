
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()

        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {self.root}"
            )

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"Expected a dataset folder: {self.root}"
            )

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        self.files = sorted([
            p for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ])

        if not self.files:
            print(f"Warning: No images found in {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def get_transform(size, crop, final_size):
    transform_list = []

    if size > 0:
        transform_list.append(transforms.Resize(size))

    if crop:
        transform_list.append(
            transforms.RandomCrop(final_size)
        )
    else:
        transform_list.append(
            transforms.Resize((final_size, final_size))
        )

    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def calc_mean_std(feat, eps=1e-5):
    # Input shape: [batch_size, channels, height, width]
    if feat.dim() != 4:
        raise ValueError(
            f"Expected a 4D tensor, got shape {tuple(feat.shape)}"
        )

    batch_size, channels = feat.shape[:2]

    feat_var, feat_mean = torch.var_mean(
        feat.reshape(batch_size, channels, -1),
        dim=2,
        unbiased=False,
        keepdim=True
    )

    feat_mean = feat_mean.reshape(
        batch_size, channels, 1, 1
    )
    feat_std = torch.sqrt(
        feat_var + eps
    ).reshape(batch_size, channels, 1, 1)

    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    # Content and style features: [B, C, H, W]
    if content_feat.shape[:2] != style_feat.shape[:2]:
        raise ValueError(
            "Content and style feature batches must have matching "
            "batch size and channel count."
        )

    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)

    normalized_content = (
        content_feat - content_mean
    ) / content_std

    return normalized_content * style_std + style_mean
