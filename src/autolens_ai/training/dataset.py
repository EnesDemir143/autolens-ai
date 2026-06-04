"""PyTorch Dataset for AutoLens AI training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import torch
from PIL import Image
from torch.utils.data import Dataset

from autolens_ai.data.labels import TARGET_CLASSES


class AutoLensDataset(Dataset):
    """Dataset that loads images from Phase 2 split CSVs.

    Requirements: TRN-06, TRN-07
    """

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        transform: Any = None,
        split_filter: str | None = None,
    ):
        """Initialize dataset from Phase 2 split CSV.

        Args:
            csv_path: Path to split CSV (train.csv, val.csv, internal_test.csv, or all_splits.csv)
            root_dir: Root directory containing the dataset images
            transform: Optional transform to apply to images
            split_filter: If using all_splits.csv, filter by 'train', 'val', or 'internal_test'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Load CSV
        df = pd.read_csv(csv_path)

        # Filter by split if needed
        if split_filter:
            df = df[df["split"] == split_filter]

        # Keep only accepted images
        df = df[df["mapping_status"] == "accepted"]

        # Build class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Store image paths and labels
        self.samples = []
        for _, row in df.iterrows():
            img_path = self.root_dir / row["relative_path"]
            label = row["normalized_label"]
            if label in self.class_to_idx:
                self.samples.append((img_path, self.class_to_idx[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transform
        if self.transform:
            image = self.transform(image)  # type: ignore[assignment]

        return image, label  # type: ignore[return-value]

    def get_class_counts(self) -> dict[str, int]:
        """Return class distribution for computing weights."""
        counts = {cls: 0 for cls in TARGET_CLASSES}
        for _, label_idx in self.samples:
            cls_name = self.idx_to_class[label_idx]
            counts[cls_name] += 1
        return counts
