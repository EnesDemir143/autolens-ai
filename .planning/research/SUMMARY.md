# Research Summary

## Stack

Use uv + Python 3.12 with PyTorch/Lightning/timm/transformers for training and Gradio for UI. The overall comparison has four model families: MobileNetV4 Conv Medium, EfficientNet-B2, ResNet18, and DINOv3 ViT-S/16. DINOv3 is the intended main model path and must run non-LoRA before optional LoRA. Keep ONNX optional until a working PyTorch inference path exists. Use the package list from `docs/plan.md` plus `transformers` for DINOv3.

## Dataset Sources to Investigate

Primary Kaggle candidates:
- `ademboukhris/cars-body-type-cropped` — 7 body classes, about 7,000 images, useful for SUV/VAN/HATCHBACK/PICK-UP/SEDAN but missing station wagon, micro, F1/open-wheel.
- `mayurmahurkar/stanford-car-body-type-data` — assignment-referenced Stanford-derived body type data.
- `sujaykapadnis/vehicle-type-image-dataset` — sedan, pickup, SUV, hatchback and other vehicle classes.
- `lyensoetanto/vehicle-images-dataset` — 15,645 images in vehicle categories including city car, sedan, SUV, van, truck.
- `eimadevyni/car-model-variants-and-images-dataset` — large image + metadata source with body type metadata; useful if class filtering is needed.

Primary Hugging Face candidates:
- `DrBimmer/vehicle-classification` — image classification dataset, 2.7k rows, vehicle categories including type/make labels.
- `kitrofimov/cbsc` — Car Body Style Classification, 595 rows, labels include SUV, van, pickup, sedan, hatchback and others.
- `rebrowser/carfax-dataset` and `rebrowser/carscom-dataset` — large metadata datasets with bodyStyle distributions; useful for research/label strategy, not necessarily image training.

Critical gap: open-wheel/F1, station wagon, micro, and clean pickup may require targeted collection or additional sources beyond the assignment reference datasets. Final dataset target is 36k–40k raw candidates and about 32k clean images (roughly 4k/class); ambiguous `City Car` and generic `Truck` labels must not be mapped directly without filtering.

## Watch Outs

The project should not assume one public dataset exactly matches the 8 required labels. The roadmap must include an explicit dataset curation/audit phase before model training.
