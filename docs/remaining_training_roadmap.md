# Training Roadmap — Remaining Experiments

**Tarih:** 2026-05-09  
**Hedef:** En iyi modeli seçip tek bir final run ile optimize etmek.

---

## Genel Strateji

Her model **standart CrossEntropy** ile çalıştırılır ve karşılaştırılır.  
Final run sadece **1 kazanan model** üzerinde yapılır — her modele ayrı ayrı focal loss uygulamak gereksiz zaman kaybı.

---

## Adım 1 — Kalan Baseline Çalışmalar

Aşağıdaki modeller standart CrossEntropy + mevcut config ile çalıştırılacak:

| Model | Yöntem | Durum |
|-------|--------|-------|
| EfficientNet-B2 | baseline | ✅ Tamamlandı |
| ResNet18 | baseline | ✅ Tamamlandı |
| MobileNetV4 Conv Medium | baseline | ✅ Tamamlandı |
| **DINOv3 ViT-S (non-LoRA)** | baseline | ⏳ Bekliyor |
| **DINOv3 ViT-S (LoRA)** | baseline | ⏳ Bekliyor |

---

## Adım 2 — Model Karşılaştırma

5 modelin test seti metrikleri kıyaslanır:

| Metrik | Öncelik |
|--------|---------|
| F1-macro | 🥇 Birincil |
| Accuracy | 🥈 İkincil |
| Latency (ms) | 🥉 Deployability |
| Model boyutu | < 95 MB zorunlu |

**→ 1 kazanan seçilir.**

---

## Adım 3 — Final Run (Kazanan Model)

Kazanan model **2 ayrı run** ile optimize edilir:

### Run A — Weighted CrossEntropy + Augmentation + EMA
```yaml
loss: cross_entropy
class_weights: inverse_frequency  # dengesiz sınıflara ağırlık
augmentation: strong               # RandAugment / albumentations
ema: true                          # Exponential Moving Average
```

### Run B — Focal Loss + Augmentation + EMA
```yaml
loss: focal_loss
focal_gamma: 2.0
focal_alpha: inverse_frequency     # sınıf ağırlıkları focal'a entegre
augmentation: strong
ema: true
```

**→ Run A ve Run B karşılaştırılır, F1-macro'ya göre en iyi seçilir.**

> **Not:** EMA genellikle training'i gürültüden korur ve generalizasyonu artırır.  
> Weighted CE vs Focal: Focal az örnekli sınıfları daha agresif cezalandırır;  
> eğer class imbalance belirginse Focal daha avantajlı olabilir.

---

## Adım 4 — Export + Calibration

```bash
make checkpoint-to-safetensors  # Lightning ckpt → safetensors
make export-model               # safetensors → ONNX → simplified ONNX
make size-check                 # < 95 MB kontrolü
make calibrate-model            # temperature scaling (val seti, test dokunulmaz)
make prepare-demo-artifact      # active_model.json güncelle
```

---

## Adım 5 — UI Güncelle

`artifacts/demo/active_model.json` pointerı yeni modeli gösterecek şekilde güncellenir.  
FastAPI `/api/models` endpoint'i otomatik yeni modeli listeler.

---

## Özet Sıra

```
DINOv3 non-LoRA baseline
    ↓
DINOv3 LoRA baseline
    ↓
5 model karşılaştır → 1 kazanan
    ↓
Kazanan: Run A (Weighted CE + Aug + EMA)
Kazanan: Run B (Focal Loss + Aug + EMA)
    ↓
En iyi run → export → calibration
    ↓
UI aktif model güncelle
```
