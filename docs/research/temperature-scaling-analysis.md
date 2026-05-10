# Temperature Scaling — Teknik Analizi

> **Proje:** AutoLens AI — 8 Sınıf Araç Gövde Tipi Sınıflandırması  
> **Tarih:** 2026-05-10  
> **Bağlam:** Post-hoc olasılık kalibrasyonu için en mantıklı yöntem seçimi

---

## Neden Temperature Scaling?

Dokümandaki 9 farklı kalibrasyon yöntemi karşılaştırmasında Temperature Scaling, projenin **tüm kısıtlamaları** bir arada karşılayan tek yöntem:

| Kriter | Temperature Scaling | Diğer Yöntemler |
|---|---|---|
| Macro F1'yi korur? | ✅ Evet (kesin) | ❌ Genel olarak değiştirir |
| 3,157 sample üzerinde overfit riski? | ✅ Yok (1 parametre) | ⚠️–❌ Orta-yüksek |
| ONNX Runtime uyumluluğu? | ✅ Tek `Div` node | ❌–⚠️ Adımlı fonksiyonlar |
| 95 MB boyut sınırı? | ✅ +4 byte | ⚠️–❌ Artık parametreler |
| Enference latency etkisi? | ✅ Sıfır | ⚠️ Artışlar |

---

## Matematiksel Temel

EfficientNet-B2'nin çıktı logit vektörü **z**'yi sıcaklık parametresi **T** ile bölerek:

$$\hat{q}_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{K} \exp(z_j / T)}$$

- T > 1 → dağılım yumuşatılır (overconfidence azalır)
- T < 1 → dağılım sharpen edilir
- **Karar sınırları değişmez** → argmax aynı kalır → F1/Recall/Precision etkilenmez

## Mevcut Sonuçlar (Baseline'da Uygulanmış)

| Metrik | Önceden | Sonra |
|---|---|---|
| Accuracy | 92.39% | 92.39% (değişmedi) |
| NLL | 0.29796 | **0.24160** |
| ECE | 0.03823 | **0.01043** |

Bu sonuçlar, modelin yapısal overconfidence sorununu doğrulamaktadır.

## Avantajları

1. **Sıfır eğitim maliyeti** — Sadece L-BFGS ile 1 skaler optimizasyon
2. **Overfitting imkansı sıfır** — 1 parametre, 3.157 sample'a göre kapasitesi ihmal edilebilir düzeyde
3. **ONNX'te kusursuz export** — `Div` node, mobil/edge uyumluluğu mükemmel
4. **Gradio UI ile doğrudan uyumlu** — Çıktı tensor'ını doğrudan 8 sınıf etiketine map edebilir
5. **Açıklama kolaylığı** — Sunumda "model X kat daha güvenili çıktı, T=1.78 ile yumuşattık" diyebilirsin

## Dezavantajı

- **Global düzeltme:** Tüm 8 sınıf üzerinde tek bir skalar uygular. Eğer belirli sınıflar (örn. STATION WAGON) sistematik olarak farklı overconfidence/underconfidence gösteriyorsa, bu yöntem bunu düzeltemez.
- **Çözüm:** Eğer Classwise ECE analizi sınıf bazlı ciddi kalibrasyon sapmaları ortaya koyarsa, Dirichlet Calibration (2. öneri) denenebilir.

## Uygulama

```python
class TemperatureScaledEfficientNet(nn.Module):
    def __init__(self, base_model, temperature_val=1.78):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('temperature', torch.tensor([temperature_val]))

    def forward(self, x):
        logits = self.base_model(x)
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=1)
```

## Sonuç

**AutoLens AI için en mantıklı ve en güvenli post-hoc kalibrasyon yöntemi Temperature Scaling'dir.** Proje kısıtlamaları (95 MB limiti, ONNX uyumluluğu, F1 metriğinin korunması, küçük validasyon seti) ile mükemmel örtüşmektedir. Öncelikle bu yöntem uygulanmalı; sınıf bazlı kalibrasyon sorunu ortaya çıkarsa Dirichlet Calibration ikinci seçenek olarak değerlendirilebilir.