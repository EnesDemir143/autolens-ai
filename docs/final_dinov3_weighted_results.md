# AutoLens AI - DINOv3 Weighted Loss Model Raporu

Bu rapor, projedeki en başarılı model olan **DINOv3 (Weighted Loss + Augmentation)** modelinin eğitim, doğrulama, kalibrasyon ve test metriklerini, görsel çıktılarıyla birlikte özetlemektedir.

## 1. Model ve Eğitim Özeti
- **Base Model:** `vit_small_patch16_dinov3` (Timm)
- **Kayıp Fonksiyonu (Loss):** Weighted Cross Entropy + Label Smoothing (0.05)
- **Optimizasyon:** AdamW (LR: 0.0001, Weight Decay: 0.0001)
- **Epoch / Step:** 29 Epoch / 5940 Step
- **Görüntü İşleme:** 256x256 Resize, 256x256 Crop, RGB, Eğitim İçi Augmentasyonlar

---

## 2. Eğitim Süreci ve Grafikler

Aşağıdaki grafikler, modelin 29 epoch boyunca eğitim ve doğrulama (validation) setlerindeki kayıp (loss) ve doğruluk (accuracy) değişimlerini göstermektedir.

### Eğitim ve Doğrulama Kaybı (Training & Validation Loss)
![Training Loss](artifacts/export/dinov3_safe_weighted_latest/training_loss.png)

> **Kayıp (Loss) Değerlerinin Kıyaslanması Üzerine Not:** 
> DINOv3 modelinin bir başka varyantında (Focal Loss) kayıp değerleri 0.5 seviyelerinde gezinirken, bu modelde (Weighted Cross Entropy) 0.7 seviyelerinde görünmektedir. Ancak **bu iki loss değeri doğrudan birbiriyle kıyaslanamaz.** Focal Loss formülü gereği iyi öğrenilmiş sınıflardaki kaybı sıfıra yaklaştırdığı için her zaman sayısal olarak daha düşük çıkar. Çapraz modeller arası başarı kıyaslaması Loss değerleri üzerinden değil, aşağıdaki Accuracy ve F1 skorları üzerinden yapılmalıdır.

### Eğitim ve Doğrulama Doğruluğu (Training & Validation Accuracy)
![Training Accuracy](artifacts/export/dinov3_safe_weighted_latest/training_accuracy.png)

Grafiklerden görüleceği üzere, modelin validation accuracy değeri hızlı bir şekilde yükselmiş ve istikrarlı bir noktaya (yaklaşık ~%96) oturmuştur. Overfitting (aşırı öğrenme) belirtisi (validation loss'un aniden fırlaması gibi) gözlemlenmemiştir.

---

## 3. İç Test Seti (Internal Test) Performansı
Modelin 3170 fotoğraftan oluşan test setindeki genel metrikleri (`internal_test_results.json`):

- **Doğruluk (Accuracy):** %95.84
- **Makro F1 (Macro F1):** 0.9507
- **Ağırlıklı F1 (Weighted F1):** 0.9584

### Sınıf Bazlı Performans (Kalibrasyon Öncesi)
| Sınıf | Precision | Recall | F1-Score | Support (Örnek) |
|:---|:---:|:---:|:---:|:---:|
| **SUV** | 0.9576 | 0.9433 | 0.9504 | 670 |
| **VAN** | 0.9933 | 0.9780 | 0.9856 | 455 |
| **STATION WAGON** | 0.9524 | 0.9091 | 0.9302 | 66 |
| **MICRO** | 1.0000 | 0.9545 | 0.9767 | 22 |
| **OPEN WHEEL / F1** | 1.0000 | 0.9983 | 0.9991 | 586 |
| **SEDAN** | 0.9496 | 0.9689 | 0.9591 | 836 |
| **HATCHBACK** | 0.8667 | 0.8797 | 0.8731 | 266 |
| **PICK UP** | 0.9296 | 0.9331 | 0.9314 | 269 |

> **Weighted Loss'un Etkisi:** Az sayıda örneğe sahip **Station Wagon** (66 örnek) ve **Micro** (22 örnek) gibi sınıflarda F1 skorları %93'ün üzerine çıkmıştır. Weighted Loss, dengesiz veri setinde azınlık sınıfların da başarıyla öğrenilmesini sağlamıştır.

### Karmaşıklık Matrisi (Confusion Matrix)
![Confusion Matrix](artifacts/export/dinov3_safe_weighted_latest/confusion_matrix.png)

Hata matrisinde köşegenin (diagonal) belirgin şekilde parlak olması, modelin sınıfları oldukça iyi ayırabildiğini göstermektedir. Görülen az sayıdaki hatalar, görsel olarak birbirine çok benzeyen Hatchback/SUV ve SUV/Sedan arasında yaşanmıştır.

---

## 4. Olasılık Kalibrasyonu (Dirichlet Calibration)
Modelin sınıflandırma sonrası ürettiği güven (confidence) oranlarının ne kadar gerçeği yansıttığını iyileştirmek için Grid Search ile kalibrasyon uygulanmıştır. En başarılı yöntem **Dirichlet Calibration (ODIR $\lambda=0.001$)** seçilmiştir.

### Test Seti (Internal Test Split) Kalibrasyon Değişimi
| Metrik | Kalibrasyon Öncesi | Dirichlet Kalibrasyon Sonrası | Gelişim |
|---|---|---|---|
| **Accuracy** | 95.77% | **95.99%** | + 0.22% |
| **NLL (Loss)** | 0.2644 | **0.1487** | Neredeyse yarı yarıya düştü |
| **ECE** | 0.0984 | **0.0097** | Hata %9.8'den <%1'e indi |
| **Class-wise ECE** | 0.0291 | **0.0045** | Çok daha tutarlı olasılıklar |

Kalibrasyon sonrası model, "Ben %90 eminim" dediğinde gerçekten ortalama %90 ihtimalle doğru bilir hale getirilmiş ve Expected Calibration Error (ECE) değeri mükemmele (%0.009) yaklaşmıştır.

---

## 5. Çıktı Dosyaları ve Model Boyutları
Proje şartnamesinde (Yazlab 2 Proje 3) istenen model boyutunun maksimum 95 MB olması kriteri başarıyla sağlanmıştır:

- **ONNX Model Boyutu:** 82.6 MB
- **Safetensors Boyutu:** 82.4 MB

Model, herhangi bir ekstra sıkıştırma (Quantization) işlemine gerek kalmadan proje sınırları dahilindedir.
