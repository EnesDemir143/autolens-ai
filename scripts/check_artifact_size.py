#!/usr/bin/env python
"""Check arbitrary model artifact files against the 95 MB limit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-size", type=float, default=95.0)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    results = []
    ok = True
    for path in args.paths:
        if not path.exists():
            results.append({"path": str(path), "exists": False, "under_limit": False})
            ok = False
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        under = size_mb <= args.max_size
        ok = ok and under
        results.append(
            {"path": str(path), "exists": True, "size_mb": round(size_mb, 3), "under_limit": under}
        )
        status = "✓" if under else "✗"
        print(f"{status} {path}: {size_mb:.2f} MB / {args.max_size:.2f} MB")

    payload = {"max_size_mb": args.max_size, "results": results, "all_under_limit": ok}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
