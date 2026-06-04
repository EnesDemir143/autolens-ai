# AutoLens AI — Eğitim Notları

Bu doküman, AutoLens AI projesinde kullanılan eğitim ayarlarını ve denenen model/loss/augmentation stratejilerini özetler.

## Kullanılan Araçlar

- PyTorch Lightning
- Hydra/YAML config sistemi
- timm model factory
- Albumentations
- Torchmetrics + scikit-learn
- ONNX Runtime export/inference
- Weights & Biases opsiyonel logging

## Genel Eğitim Ayarları

| Ayar | Değer |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Early Stopping | patience=10 |
| Loss | Cross Entropy / Weighted CE / Focal |
| Label Smoothing | 0.05 |
| Device | MPS / CPU fallback |
| Image Size | 256 (DINOv3), 224 (CNN) |

## Denenen Modeller

| Model | Config |
|---|---|
| DINOv3 ViT-S/16 weighted | `configs/experiments/dinov3_safe_weighted_aug.yaml` |
| DINOv3 ViT-S/16 focal | `configs/experiments/dinov3_safe_focal_aug.yaml` |
| DINOv3 ViT-S/16 baseline | `configs/experiments/baseline_0_dinov3_vits16.yaml` |
| EfficientNet-B2 | `configs/experiments/baseline_0_efficientnet_b2.yaml` |
| ResNet18 | `configs/experiments/baseline_0_resnet18.yaml` |
| MobileNetV4 | `configs/experiments/baseline_0_mobilenetv4.yaml` |

## Augmentation

Safe augmentation stratejisi araç gövde tipini bozmayan hafif dönüşümlerden oluşur:

- Horizontal flip
- Küçük rotation
- Hafif brightness/contrast değişimi
- Saturation/hue değişimi

Aşırı güçlü augmentation, araç tipini ayırt eden görsel ipuçlarını bozabildiği için final modelde kullanılmadı.

## Loss Seçimi

Dengesiz sınıf dağılımı nedeniyle weighted loss önemliydi. Final modelde:

- Weighted Cross Entropy
- Label Smoothing 0.05
- EMA
- Safe augmentation

kombinasyonu kullanıldı.

## Export Akışı

```bash
make checkpoint-to-safetensors
make export-model
make size-check
make calibrate-dinov3-all
make prepare-demo-artifact
```

Bu akış sonunda model `model.safetensors`, `model.onnx`, metadata, metrics ve calibration artifactleri oluşturulur.

## Sonuç

DINOv3 ViT-S/16 weighted modeli, hem macro F1 hem de accuracy açısından en iyi sonucu verdi ve 95 MB model boyutu şartını sağladı.