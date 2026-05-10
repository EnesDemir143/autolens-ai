# Model Karşılaştırma — Baseline Sonuçları

**Tarih:** 2026-05-10  
**Test seti:** `artifacts/dataset/splits/internal_test.csv` (3170 sample)  
**Metrik önceliği:** F1-macro > Accuracy > Model boyutu

---

## Genel Karşılaştırma

| Model | Params | Size (safetensors) | Test Accuracy | F1-macro | F1-weighted | Test Loss |
|---|---:|---:|---:|---:|---:|---:|
| **DINOv3 ViT-S/16** | 21.6 M | 82.4 MB | **0.9438** | **0.9187** | **0.9442** | — |
| EfficientNet-B2 | ~9 M | 29.7 MB | 0.9202 | 0.9004 | 0.9205 | 0.284 |
| ResNet18 | ~11 M | 42.7 MB | 0.8968 | 0.8583 | 0.8963 | 0.523 |
| MobileNetV4-Conv-M | ~9 M | 32.5 MB | 0.8905 | 0.8505 | 0.8905 | 0.591 |

> DINOv3 test metrikleri `run_internal_test.py` ile safetensors üzerinden hesaplandı (Lightning test loop dışı).  
> EfficientNet-B2, ResNet18, MobileNetV4 metrikleri `metadata.json → metrics_summary` kaynağından (Lightning test loop).  
> Tüm modeller aynı internal test seti üzerinde değerlendirildi (3170 sample, veri sızıntısı yok).

---

## Per-Class Metrikler

### DINOv3 ViT-S/16 (test acc: 0.9438 | F1-macro: 0.9187)

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUV | 0.9666 | 0.9060 | 0.9353 | 670 |
| VAN | 0.9759 | 0.9780 | 0.9769 | 455 |
| STATION WAGON | 0.9474 | 0.8182 | 0.8780 | 66 |
| MICRO | 0.9000 | 0.8182 | 0.8571 | 22 |
| OPEN WHEEL / F1 | 0.9983 | 1.0000 | 0.9991 | 586 |
| SEDAN | 0.9385 | 0.9486 | 0.9435 | 836 |
| HATCHBACK | 0.7945 | 0.8722 | 0.8315 | 266 |
| PICK UP | 0.9018 | 0.9554 | 0.9278 | 269 |
| **macro avg** | **0.9279** | **0.9121** | **0.9187** | 3170 |
| **weighted avg** | **0.9456** | **0.9438** | **0.9442** | 3170 |

### EfficientNet-B2 (test acc: 0.9202 | F1-macro: 0.9004)

| Sınıf | Per-class Accuracy | Support |
|---|---:|---:|
| SUV | 0.9090 | 670 |
| VAN | 0.9824 | 455 |
| STATION WAGON | 0.7424 | 66 |
| MICRO | 0.9545 | 22 |
| OPEN WHEEL / F1 | 0.9846 | 586 |
| SEDAN | 0.9282 | 836 |
| HATCHBACK | 0.8008 | 266 |
| PICK UP | 0.8364 | 269 |

### ResNet18 (test acc: 0.8968 | F1-macro: 0.8583)

| Sınıf | Per-class Accuracy | Support |
|---|---:|---:|
| SUV | 0.9104 | 670 |
| VAN | 0.9165 | 455 |
| STATION WAGON | 0.6212 | 66 |
| MICRO | 0.7727 | 22 |
| OPEN WHEEL / F1 | 0.9949 | 586 |
| SEDAN | 0.9031 | 836 |
| HATCHBACK | 0.7444 | 266 |
| PICK UP | 0.8253 | 269 |

### MobileNetV4-Conv-Medium (test acc: 0.8905 | F1-macro: 0.8505)

| Sınıf | Per-class Accuracy | Support |
|---|---:|---:|
| SUV | 0.8284 | 670 |
| VAN | 0.9495 | 455 |
| STATION WAGON | 0.7121 | 66 |
| MICRO | 0.8636 | 22 |
| OPEN WHEEL / F1 | 0.9863 | 586 |
| SEDAN | 0.9043 | 836 |
| HATCHBACK | 0.7368 | 266 |
| PICK UP | 0.8922 | 269 |

---

## Zor Sınıflar Karşılaştırması

| Sınıf | DINOv3 | EfficientNet-B2 | ResNet18 | MobileNetV4 |
|---|---:|---:|---:|---:|
| STATION WAGON | **0.878** (F1) | 0.742 (acc) | 0.621 (acc) | 0.712 (acc) |
| MICRO | **0.857** (F1) | 0.955 (acc) | 0.773 (acc) | 0.864 (acc) |
| HATCHBACK | **0.832** (F1) | 0.801 (acc) | 0.744 (acc) | 0.737 (acc) |
| PICK UP | **0.928** (F1) | 0.836 (acc) | 0.825 (acc) | 0.892 (acc) |

> Not: DINOv3 için F1, diğerleri için per-class accuracy kullanıldı (diğer modellerin per-class F1'i hesaplanmadı).

---

## Model Boyutu

| Model | Safetensors | ONNX | < 95 MB? |
|---|---:|---:|:---:|
| DINOv3 ViT-S/16 | 82.4 MB | 82.6 MB | ✅ |
| EfficientNet-B2 | 29.7 MB | 29.4 MB | ✅ |
| ResNet18 | 42.7 MB | 42.6 MB | ✅ |
| MobileNetV4-Conv-M | 32.5 MB | 32.1 MB | ✅ |

---

## Kalibrasyon (Temperature Scaling) — Internal Test Üzerinde

Temperature val setinde fit edildi, aşağıdaki before/after değerleri **internal test** üzerinde ölçüldü.

| Model | Temperature | ECE before | ECE after | NLL before | NLL after |
|---|---:|---:|---:|---:|---:|
| **DINOv3 ViT-S/16** | 1.7211 | 0.0415 | **0.0136** | 0.2372 | **0.1784** |
| EfficientNet-B2 | 1.7799 | 0.0393 | **0.0107** | 0.2745 | **0.2301** |
| ResNet18 | — | — | — | — | — |
| MobileNetV4-Conv-M | — | — | — | — | — |

> ResNet18 ve MobileNetV4 için calibration uygulanmadı.

---

## Karar

**Kazanan: DINOv3 ViT-S/16**

- F1-macro 0.9187 — diğer modellerin en iyisinden (EfficientNet 0.9004) +1.8 puan üstün
- Zor sınıflarda (STATION WAGON, HATCHBACK, PICK UP) belirgin fark
- 82 MB safetensors ile 95 MB limitinin altında
- ONNX export sorunu çözülmesi gerekiyor (EVA `is_causal` uyumsuzluğu)

**Sonraki adım:** DINOv3 üzerinde augmentation + EMA ile final run (Run A: Weighted CE, Run B: Focal Loss).
