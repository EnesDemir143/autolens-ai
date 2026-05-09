"""Download MICRO class datasets from Roboflow Universe."""

import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

api_key = os.environ["ROBOFLOW_API_KEY"]
rf = Roboflow(api_key=api_key)

datasets = [
    ("thesis-aarau", "thesis-n4a7t", 3, "datasets/roboflow-microcar-thesis"),
    # vemo/car-types-nvqdm has no downloadable versions, skipped
]

for workspace, project, version, location in datasets:
    print(f"\nİndiriliyor: {workspace}/{project} → {location}")
    try:
        proj = rf.workspace(workspace).project(project)
        proj.version(version).download("yolov8", location=location)
        count = sum(1 for _ in Path(location).rglob("*.jpg"))
        count += sum(1 for _ in Path(location).rglob("*.png"))
        print(f"✓ {count} görsel indirildi: {location}")
    except Exception as e:
        print(f"✗ Hata ({workspace}/{project}): {e}")
