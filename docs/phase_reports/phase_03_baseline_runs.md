# Baseline Run Sonuçları — Phase 3

**Tarih:** 2026-05-08  
**Branch:** `feat/baseline-training-pipeline`  
**Dataset:** 25,285 train / 3,157 val / 3,170 test (internal)

---

## Çalıştırılan Modeller

| | ResNet18 | MobileNetV4 Conv Medium |
|---|---|---|
| Config | `baseline_0_resnet18.yaml` | `baseline_0_mobilenetv4.yaml` |
| Checkpoint | `baseline_0_resnet18_20260508_014023` | `baseline_0_mobilenetv4_20260508_025041` |
| Augmentation | ❌ (baseline_0) | ❌ (baseline_0) |
| Pretrained | ✅ ImageNet | ✅ ImageNet |
| Optimizer | AdamW lr=0.001 | AdamW lr=0.001 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Precision | bf16-mixed | bf16-mixed |
| Early Stopping | patience=10, monitor=val/loss | patience=10, monitor=val/loss |
| Max Epochs | 100 | 100 |
| Actual Epochs | 21 (stopped at epoch 20) | 19 (stopped at epoch 18) |
| Best Epoch | 13 (val/f1_macro=0.843) | 16 (val/f1_macro=0.834) |

---

## Test Seti Sonuçları (Best Checkpoint)

| Metrik | ResNet18 | MobileNetV4 |
|---|---|---|
| **Accuracy** | 87.10% | 88.04% |
| **F1 Macro** | **82.69%** | **82.49%** |
| **F1 Weighted** | 87.01% | 87.91% |
| Precision (macro) | 87.60% | 83.33% |
| Recall (macro) | 79.84% | 81.81% |

> Not: Yukarıdaki değerler `generate_plots.py` ile internal_test.csv üzerinden hesaplanmıştır (3,170 sample).  
> wandb summary'deki değerler farklı test seti boyutundan dolayı hafif farklı görünebilir.

---

## Per-Class F1 Skorları

| Sınıf | ResNet18 F1 | MobileNetV4 F1 | Test Support |
|---|---|---|---|
| SUV | 0.8638 | 0.8648 | 670 |
| VAN | 0.9303 | 0.9463 | 455 |
| STATION WAGON | 0.6200 | 0.6094 | 66 |
| MICRO | 0.8500 | 0.7907 | 22 |
| OPEN WHEEL / F1 | 0.9731 | 0.9822 | 586 |
| SEDAN | 0.8617 | 0.8793 | 836 |
| HATCHBACK | 0.6692 | 0.6640 | 266 |
| PICK UP | 0.8471 | 0.8624 | 269 |

**Gözlemler:**
- STATION WAGON ve HATCHBACK en zayıf sınıflar — az test sample + az train verisi
- OPEN WHEEL / F1 ve VAN her iki modelde de çok güçlü
- MICRO'da ResNet18 daha iyi (0.85 vs 0.79), muhtemelen az sample'da daha iyi genelliyor

---

## Overfitting Durumu

| | ResNet18 | MobileNetV4 |
|---|---|---|
| Train Acc (son epoch) | ~99.2% | ~98.1% |
| Val Acc (best epoch) | 89.0% | 87.2% |
| Train/Val gap | ~10 pp | ~11 pp |

Her iki modelde de belirgin overfitting var. Augmentation (Phase 3 baseline_1+) ve regularization ile azaltılacak.

---

## Model Boyutları

| | Weights Only | .ckpt (optimizer dahil) | 95 MB Limit |
|---|---|---|---|
| ResNet18 | **44.8 MB** | 128 MB | ✅ |
| MobileNetV4 | **34.1 MB** | 97 MB | ✅ |

SafeTensors formatına export edildiğinde her iki model de 95 MB limitinin altında.

---

## Üretilen Artifacts

Her checkpoint klasöründe:
- `confusion_matrix.png` — 8x8 normalized heatmap
- `per_class_metrics.txt` — sınıf bazlı precision/recall/f1/support
- `training_loss.png` — train & val loss (epoch bazlı)
- `training_accuracy.png` — train & val accuracy (epoch bazlı)

---

## Sonraki Adımlar (Phase 4)

- [ ] EfficientNet-B2 baseline run
- [ ] DINOv3 ViT-S/16 (ana model adayı)
- [ ] Augmentation açık run'lar (baseline_1+)
- [ ] 4 model karşılaştırması: ResNet18 vs MobileNetV4 vs EfficientNet-B2 vs DINOv3
- [ ] Final model seçimi (macro F1 öncelikli)
- [ ] SafeTensors export
