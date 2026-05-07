"""Lightning DataModule for AutoLens AI training.

Requirements: TRN-06, TRN-07
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from autolens_ai.data.labels import TARGET_CLASSES
from autolens_ai.training.dataset import AutoLensDataset
from autolens_ai.training.preprocessing import PreprocessConfig


class AutoLensDataModule(pl.LightningDataModule):
    """Lightning DataModule for AutoLens AI.
    
    Consumes Phase 2 split CSVs and provides train/val/test dataloaders.
    """

    def __init__(
        self,
        data_root: str | Path,
        train_csv: str | Path,
        val_csv: str | Path,
        test_csv: str | Path,
        preprocess_config: PreprocessConfig,
        batch_size: int = 32,
        num_workers: int = 4,
        use_augmentation: bool = False,
        use_weighted_sampler: bool = False,
        class_weights: dict[str, float] | None = None,
    ):
        """Initialize DataModule.
        
        Args:
            data_root: Root directory containing dataset images
            train_csv: Path to train.csv
            val_csv: Path to val.csv
            test_csv: Path to internal_test.csv
            preprocess_config: Preprocessing configuration
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            use_augmentation: If True, apply augmentation to training (Baseline 1+)
            use_weighted_sampler: If True, use WeightedRandomSampler (Baseline 2)
            class_weights: Optional class weights for loss computation
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.test_csv = Path(test_csv)
        self.preprocess_config = preprocess_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation
        self.use_weighted_sampler = use_weighted_sampler
        self.class_weights = class_weights
        
        self.train_dataset: AutoLensDataset | None = None
        self.val_dataset: AutoLensDataset | None = None
        self.test_dataset: AutoLensDataset | None = None
    
    def setup(self, stage: str | None = None) -> None:
        """Setup datasets."""
        if stage == "fit" or stage is None:
            # Training transform
            train_transform = self.preprocess_config.get_train_transform(augment=self.use_augmentation)
            self.train_dataset = AutoLensDataset(
                csv_path=self.train_csv,
                root_dir=self.data_root,
                transform=train_transform,
            )
            
            # Validation transform (always deterministic)
            val_transform = self.preprocess_config.get_val_transform()
            self.val_dataset = AutoLensDataset(
                csv_path=self.val_csv,
                root_dir=self.data_root,
                transform=val_transform,
            )
        
        if stage == "test" or stage is None:
            test_transform = self.preprocess_config.get_val_transform()
            self.test_dataset = AutoLensDataset(
                csv_path=self.test_csv,
                root_dir=self.data_root,
                transform=test_transform,
            )
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        assert self.train_dataset is not None
        
        sampler = None
        shuffle = True
        
        if self.use_weighted_sampler:
            # Compute sample weights for WeightedRandomSampler
            class_counts = self.train_dataset.get_class_counts()
            total = sum(class_counts.values())
            
            # Inverse frequency weights
            class_weight_map = {
                cls: total / (len(TARGET_CLASSES) * count) if count > 0 else 0.0
                for cls, count in class_counts.items()
            }
            
            # Build sample weights
            sample_weights = []
            for _, label_idx in self.train_dataset.samples:
                cls_name = self.train_dataset.idx_to_class[label_idx]
                sample_weights.append(class_weight_map[cls_name])
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False  # Sampler handles shuffling
        
        # Create generator for reproducibility
        generator = torch.Generator()
        generator.manual_seed(42)  # Will be set by pl.seed_everything
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=generator,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    @staticmethod
    def compute_class_weights(
        class_counts: dict[str, int],
        max_weight: float = 5.0,
    ) -> torch.Tensor:
        """Compute class weights for loss function.
        
        Args:
            class_counts: Dictionary mapping class names to counts
            max_weight: Maximum weight cap to prevent gradient instability
            
        Returns:
            Tensor of class weights in TARGET_CLASSES order
        """
        total = sum(class_counts.values())
        weights = []
        
        for cls in TARGET_CLASSES:
            count = class_counts.get(cls, 0)
            if count > 0:
                weight = total / (len(TARGET_CLASSES) * count)
                weight = min(weight, max_weight)  # Cap weight
            else:
                weight = 0.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
