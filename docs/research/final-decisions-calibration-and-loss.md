# AutoLens AI — Final Karar: Kalibrasyon ve Loss Seçimi

> **Tarih:** 2026-05-10  
> **Karar verenler:** Proje ekibi  
> **Durum:** ✅ Onaylandı — Uygulanacak

---

## Özet

İki ayrı araştırma dokümanından elde ettiğimiz bulgular ışığında, AutoLens AI projesi için iki teknik kararı verdik:

| Karar | Seçilen Teknik | Sebep |
|---|---|---|
| **1. Post-hoc Kalibrasyon** | **Temperature Scaling** | F1'i korumaz, ONNX uyumluluğu, sıfır overfit riski |
| **2. Loss Fonksiyonu** | **Class-Weighted CE** | Macro F1 +2.05pp artış, zor sınıfları daha iyi öğreniyor |

---

## Karar 1: Temperature Scaling (Post-Hoc Kalibrasyon)

### Neden?

Mevcut EfficientNet-B2 baseline'umuz (Accuracy: 92.39%) zaten iyi çalışıyor, ancak çıktı olasılıkları overconfident. Temperature Scaling:

- ✅ **Macro F1, Precision, Recall'i korumaz** — karar sınırlarını değiştirmez
- ✅ **NLL'yi 0.297 → 0.241'e düşürdü** (zaten kanıtlanmış)
- ✅ **ECE'yi 0.038 → 0.010'a düşürdü**
- ✅ **1 parametre** → 3,157 sample üzerinde overfit riski yok
- ✅ **ONNX'te tek `Div` node** → 95 MB sınırı etkilemez, latency artışı sıfır
- ✅ Gradio UI ile doğrudan uyumlu

### Neden Değil?

- Diğer yöntemler (Dirichlet, Vector Scaling) sınıf bazlı kalibrasyon sunar ama:
  - Decision boundary'leri değiştirir → F1 riski
  - 72 parametre → küçük validasyon setinde overfit riski
  - Temperature Scaling zaten ECE'yi 0.010'a düşürmüş → yeterince iyi

### Eylem

```
Model eğitimi: Aynı (Temperature Scaling eğitim değil, sonrası uygulanacak)
Sıcaklık optimizasyonu: L-BFGS ile validasyon seti üzerinde 1 parametre
Export: TemperatureScaledEfficientNet → ONNX → Gradio
```

---

## Karar 2: Class-Weighted CE (Loss Fonksiyonu)

### Neden?

DINOv3 ViT-S/16 backbone'unda yapılan ablation:

- **F1-macro:** 0.9302 → **0.9507** (+2.05pp) ✅
- **Accuracy:** 0.9505 → **0.9584** (+0.79pp) ✅  
- **Macro Recall:** 0.9269 → **0.9456** (+1.87pp) ✅

En kritik kazanım **az örnekli sınıflarda**:
- STATION WAGON: F1 0.8652 → 0.9302 (**+6.5pp**)
- MICRO: F1 0.9302 → 0.9767 (**+4.65pp**)
- PICK UP: F1 0.9070 → 0.9314 (**+2.44pp**)

### Ekstra Avantaj

- Focal loss'a göre **ekstra hiperparametre yok** (gamma, alpha ayarı gerekmez)
- Sadece sınıf frekanslarının tersine orantılı ağırlıklar hesapla
- Pipeline basitleşir, tekrarlanabilirlik artar

### Neden Focal Loss Değil?

- Bu veri setinde ve bu pipeline'da weighted CE'nin üstü geçemedi
- Focal loss'un easy-example suppression efekti, az veri olan sınıfları dezavantajlıyor
- Batch size farkı (64 vs 128) karşılaştırma güçlüğü yaratıyor; weighted CE büyük batch ile daha stabil

---

## Birlikte Nasıl Çalışırlar?

Bu iki teknik **birbirini tamamlayan**, farklı aşamalarda çalışan çözümler:

```
Eğitim Aşaması:
  DINOv3 ViT-S/16 + Safe Augmentation + Class-Weighted CE
       ↓
  Eğitilmiş model (doğrudan sınıf ağırlıklarıyla öğrenilmiş)
       ↓
İnferans Aşaması:
  Girdi → DINOv3 → Logits → Temperature Scaling → Softmax → Tahmin
```

1. **Class-weighted CE** eğitim sırasında az temsil edilen sınıflara daha fazla önem verir → daha dengeli öğrenme
2. **Temperature Scaling** inferans sırasında çıktı olasılıklarını kalibre eder → güvenilir tahminler

---

## Eylem Planı

| # | Adım | Durum |
|---|---|---|
| 1 | DINOv3 modeline class-weighted CE geçişi | 🔲 Yapılacak |
| 2 | Aynı batch_size ve seed ile replicasyon deneyi | 🔲 Yapılacak |
| 3 | Internal test'te metrik karşılaştırması | 🔲 Yapılacak |
| 4 | Temperature Scaling implementasyonu (EfficientNet-B2 veya DINOv3'e) | 🔲 Yapılacak |
| 5 | ONNX export + boyut kontrolü (< 95 MB) | 🔲 Yapılacak |
| 6 | Gradio UI entegrasyonu | 🔲 Yapılacak |

---

## Riskler ve Önlemler

| Risk | Önlem |
|---|---|
| Class-weighted CE ile DINOv3 eğitimi uzun sürebilir | Büyük batch (128) kullan, mixed precision etkinleştir |
| Sınıf ağırlıkları optimizasyonu | `sklearn.utils.class_weight.compute_class_weight` ile otomatik hesapla |
| Temperature Scaling parametresi veri sızıntısı | Nested cross-validation kullan (doküman 1'deki protokol) |
| ONNX export hataları | `torch.onnx.export` ile doğrudan export et, manuel müdahale yok |

---

## Son Söz

Bu iki karar birlikte, AutoLens AI projesinin **hem doğruluk hem güvenilirlik hem de dağıtım uygunluğu** açısından en iyi sonucu verecek kombinasyondur. DINOv3 + Class-Weighted CE modelini eğitip, üzerine Temperature Scaling ile kalibrasyon eklemek, hem F1 skorunu artıracak hem de kullanıcıya güvenilir olasılık tahminleri sunacaktır.