# Training Pipeline Plan

## Genel Akış

```
Phase 1: Baseline Karşılaştırma
    ↓
Phase 2: Final Model İyileştirme
    ↓
Phase 3: Calibration
    ↓
Phase 4: Demo + Rapor
```

---

## Phase 1 — Baseline Karşılaştırma

**Amaç:** Mimari seçimi. Hangi model bu dataset üzerinde daha iyi öğreniyor?

**Koşullar (tüm modellerde eşit):**
- Augmentation: kapalı
- EMA: kapalı
- Dataset mean/std: `[0.4429, 0.4354, 0.4370]` / `[0.2456, 0.2421, 0.2449]` (train split'ten hesaplanmış)
- Class weighted loss: açık
- Seed: 42

**Çalıştırılacak 5 run:**

```bash
uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml
uv run python scripts/train.py --config configs/experiments/baseline_0_mobilenetv4.yaml
uv run python scripts/train.py --config configs/experiments/baseline_0_efficientnet_b2.yaml
uv run python scripts/train.py --config configs/experiments/baseline_0_dinov3_vits16.yaml
uv run python scripts/train.py --config configs/experiments/baseline_0_dinov3_vits16_lora.yaml --lora
```

**Karşılaştırma kriteri:** Val F1 (macro). Accuracy, Precision, Recall de raporlanır.

**Model seçimi:** En yüksek val F1. Fark çok küçükse (<1-2%) model boyutu ve inference hızı da göz önüne alınır (95 MB limiti var).

---

## Phase 4 — Final Model İyileştirme ve Kaydetme

**Amaç:** Kazanan mimariyi iyileştir, calibrate et, deploy-ready hale getir.

### 4.1 — Augmentation + EMA ile Eğitim

Sadece kazanan modele uygulanır.

Config'de açılacaklar:
```yaml
use_augmentation: true
use_ema: true
ema_decay: 0.999
```

Augmentation + EMA birlikte kullanılır çünkü EMA, augmentation'ın getirdiği training noise'unu yumuşatır — ikisi sinerjik çalışır.

> **Not:** Baseline sonuçlarında kazanan ile 2. arasındaki fark çok küçükse (<1-2% F1), o modelde de augmentation + EMA denenebilir. Bu durumda mimari seçimi daha sağlam gerekçeye oturur. Calibration ise her durumda sadece deploy edilecek final modele uygulanır.

### 4.2 — Calibration (Temperature Scaling)

- **Yöntem:** Temperature Scaling (Guo et al., 2017) — tek scalar parametre T
- **Fit:** val split üzerinde (NLL minimize ederek T öğren)
- **Evaluate:** internal_test split üzerinde (ECE, NLL raporla)
- **Neden val'da fit:** T tek parametre, 3k sample'da overfit ihtimali yok; literatür standardı

> **Not:** Temperature Scaling yerine başka bir calibration yöntemi de denenebilir (örn. Platt Scaling, Isotonic Regression, Vector Scaling). Yöntem ne olursa olsun prensip değişmez: **val'da fit, internal_test'te evaluate.** Final modele karar verilince yönteme bakılır.

### 4.3 — Model Kaydetme

- ONNX export veya `torch.save(model.state_dict())`
- Boyut kontrolü: 95 MB altında olmalı
- `.ckpt` dosyası optimizer state içerdiği için büyük görünür — sadece weights export edilince küçülür
  - ResNet18 weights: ~45 MB
  - DINOv3 ViT-S weights: ~22 MB
  - İkisi de limitin altında

---

## Phase 5 — Gradio Demo

- Final calibrated model yüklenir
- Kullanıcı görsel yükler → tahmin + confidence + 8 sınıf olasılık bar chart gösterilir
- Inference preprocessing, training preprocessing ile birebir aynı olmalı
- Latency kabul edilebilir olmalı (assignment puan kriteri)

## Phase 6 — IEEE Raporu

- Tüm 5 baseline sonuçları karşılaştırma tablosu
- Final model (augmentation + EMA + calibration) sonuçları
- Dataset kaynakları, preprocessing, mimari seçim gerekçesi
- Tüm grafikler: loss, accuracy, confusion matrix
- Minimum 4 sayfa, IEEE formatı, LaTeX kaynak + PDF teslim

---

## Kararlar ve Gerekçeler

| Karar | Gerekçe |
|---|---|
| EMA baseline'larda kapalı | Karşılaştırma temiz olsun, EMA mimari farkını maskelemesin |
| Augmentation sadece kazananda | Diğer modellerde gereksiz maliyet, metodolojik olarak doğru |
| Calibration sadece final modelde | Makale yazmıyoruz, 5 modele calibration gereksiz run |
| Val'da calibration fit | 3 split var, ayrı cal seti train'i 25k→21k düşürür, tek scalar için gerek yok |
| Ayrı LoRA config | Checkpoint ve experiment name ayrı, ileride LoRA-specific ayar kolayca değiştirilebilir |
