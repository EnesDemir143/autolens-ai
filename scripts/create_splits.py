#!/usr/bin/env python3
"""Create stratified train/val/test splits with reproducible seed.

Usage:
    uv run python scripts/create_splits.py
    uv run python scripts/create_splits.py --train 0.7 --val 0.15 --test 0.15 --seed 123
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]


def create_stratified_splits(
    input_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Counter]:
    """Create stratified splits with reproducible seed.
    
    Args:
        input_csv: Path to merged manifest CSV
        output_dir: Output directory for split CSVs
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with split counts per class
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # Load data
    df = pd.read_csv(input_csv)
    
    # Filter eligible samples
    df = df[df["split_eligible"] == "yes"].copy()
    
    print(f"Loaded {len(df)} eligible samples")
    print(f"Classes: {df['normalized_label'].nunique()}")
    print(f"Class distribution:\n{df['normalized_label'].value_counts()}")
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df["normalized_label"],
        random_state=seed,
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        stratify=temp_df["normalized_label"],
        random_state=seed,
    )
    
    # Add split column
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "internal_test"
    
    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "internal_test.csv", index=False)
    
    # Save combined
    all_df = pd.concat([train_df, val_df, test_df])
    all_df.to_csv(output_dir / "all_splits.csv", index=False)
    
    # Compute counts
    split_counts = {
        "train": Counter(train_df["normalized_label"]),
        "val": Counter(val_df["normalized_label"]),
        "internal_test": Counter(test_df["normalized_label"]),
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("Split Summary")
    print("=" * 60)
    for split_name, counts in split_counts.items():
        total = sum(counts.values())
        print(f"\n{split_name.upper()}: {total} samples")
        for label, count in sorted(counts.items()):
            pct = 100 * count / total
            print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 60)
    print(f"Seed: {seed}")
    print(f"Ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    print(f"Output: {output_dir}")
    print("=" * 60 + "\n")
    
    return split_counts


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Create stratified splits")
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/dataset/manifests/merged_manifest.csv",
        help="Input merged manifest CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/dataset/splits",
        help="Output directory for splits",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()
    
    create_stratified_splits(
        input_csv=args.input,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
