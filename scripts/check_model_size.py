#!/usr/bin/env python3
"""Check model checkpoint sizes.

Usage:
    uv run python scripts/check_model_size.py checkpoints/baseline_0_resnet18_*/best-*.ckpt
    uv run python scripts/check_model_size.py checkpoints/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def check_checkpoint_size(ckpt_path: Path, max_size_mb: float = 95.0) -> dict:
    """Check if checkpoint is under size limit."""
    size_bytes = ckpt_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    return {
        "path": str(ckpt_path),
        "size_bytes": size_bytes,
        "size_mb": size_mb,
        "size_human": format_size(size_bytes),
        "under_limit": size_mb < max_size_mb,
        "limit_mb": max_size_mb,
    }


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Check model checkpoint sizes")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Checkpoint file(s) or directory to check",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=95.0,
        help="Maximum size in MB (default: 95.0)",
    )
    args = parser.parse_args()
    
    # Collect all checkpoint files
    ckpt_files = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == ".ckpt":
            ckpt_files.append(path)
        elif path.is_dir():
            ckpt_files.extend(path.rglob("*.ckpt"))
    
    if not ckpt_files:
        print("No checkpoint files found!")
        return
    
    # Check each checkpoint
    results = []
    for ckpt_path in sorted(ckpt_files):
        result = check_checkpoint_size(ckpt_path, args.max_size)
        results.append(result)
    
    # Print results
    print("\n" + "=" * 80)
    print(f"Model Checkpoint Size Check (Limit: {args.max_size} MB)")
    print("=" * 80)
    
    for result in results:
        status = "✅ OK" if result["under_limit"] else "❌ TOO LARGE"
        print(f"\n{status}")
        print(f"  Path: {result['path']}")
        print(f"  Size: {result['size_human']} ({result['size_mb']:.2f} MB)")
        if not result["under_limit"]:
            print(f"  ⚠️  Exceeds limit by {result['size_mb'] - result['limit_mb']:.2f} MB")
    
    # Summary
    total = len(results)
    under_limit = sum(1 for r in results if r["under_limit"])
    over_limit = total - under_limit
    
    print("\n" + "=" * 80)
    print(f"Summary: {under_limit}/{total} checkpoints under {args.max_size} MB limit")
    if over_limit > 0:
        print(f"⚠️  {over_limit} checkpoint(s) exceed the limit!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
