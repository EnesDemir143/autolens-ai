# LR Finder & Batch Size Finder

PyTorch Lightning'in `Tuner` sınıfı, eğitim öncesi iki hiperparametreyi otomatik bulmak için kullanılır.

---

## Learning Rate Finder (`--find-lr`)

**Mantık:** Gerçek forward + backward pass yaparak mini bir eğitim deneyi çalıştırır. Çok küçük bir LR'den başlar, her batch'te LR'yi üstel olarak artırır ve o batch'teki loss'u kaydeder. Loss düşmeye devam ettiği sürece LR iyi demektir; loss patlamaya başladığı nokta LR'nin çok büyük olduğunu gösterir. En dik düşüşün yaşandığı nokta önerilen LR olarak seçilir. Birkaç yüz batch sonra biter ve model ağırlıkları başa sıfırlanır — asıl eğitimi bozmaz.

```
loss
 |  \
 |   \       ← burası önerilen LR
 |    \___/‾‾‾‾  ← loss patladı, LR çok büyük
 +-----------> LR (log scale)
```

**Kullanım:**
```bash
uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml --find-lr
```

Çıktı olarak `checkpoints/.../lr_finder.png` ve `tune_results.json` kaydedilir. Eğitim başlamaz, sadece öneri verilir. Önerilen LR'yi config'e yazıp normal eğitimi başlatmak gerekir.

---

## Batch Size Finder (`--find-batch-size`)

**Mantık:** Sadece forward pass yaparak (backward yok) belleğe sığan maksimum batch size'ı bulur. 2 sample ile başlar, her adımda iki katına çıkarır (2 → 4 → 8 → 16 → ...). OOM (Out of Memory) hatası aldığında bir önceki değeri önerir. Birkaç saniye sürer.

```
batch_size: 2 → 4 → 8 → 16 → 32 → OOM!
                                ↑
                          önerilen: 32
```

**Kullanım:**
```bash
uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml --find-batch-size
```

Çıktı olarak `tune_results.json` kaydedilir. Eğitim başlamaz.

---

## Önemli Notlar

- Her iki finder da eğitimi **başlatmaz**, sadece öneri üretir ve çıkar.
- LR finder birkaç epoch kadar veri geçer, batch size finder ise sadece birkaç forward pass yapar — ikisi de hızlıdır.
- Bulunan değerler config YAML'ına elle yazılmalıdır.
- MPS (Apple Silicon) üzerinde batch size finder bazen bellek tahminini yanlış yapabilir; önerilen değeri %20 düşürerek kullanmak güvenlidir.
