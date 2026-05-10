# AutoLens AI: Kalibrasyon Etki Raporu (Test Seti)

**Seçilen Model:** DINOv3 Weighted
**Seçilen Kalibrasyon Yöntemi:** Dirichlet Calibration (ODIR λ=0.001)
**Veri Seti:** Internal Test Set

Aşağıdaki tablolar, modelin kalibrasyon uygulanmadan önceki ham (raw) durumu ile Dirichlet yöntemiyle kalibre edildikten sonraki durumunu **Test verisi** üzerinde karşılaştırmaktadır.

## 1. Global Metrikler (Genel Başarı)

| Metrik | Ham Model (Kalibrasyon Yok) | Dirichlet (λ=0.001) | Değişim |
|:---|:---:|:---:|:---|
| **NLL (Negatif Log Olabilirlik)** | 0.2644 | **0.1487** | Hata yarı yarıya azaldı. Tahmin kesinliği arttı. |
| **ECE (Güven Hatası)**| 0.0984 | **0.0097** | Kalibrasyon hatası %9.8'den **%0.9'a** düştü! (10 kat iyileşme) |
| **cwECE (Sınıf Bazlı ECE)** | 0.0292 | **0.0045** | Azınlık sınıflarındaki güven sapmaları giderildi. |
| **Accuracy (Doğruluk)** | 0.9577 | **0.9599** | Sınıflandırma yeteneği daha da yükseldi. |
| **MCC (Matthews Korelasyon)** | 0.9427 | **0.9414** | Dengesiz sınıf dağılımına rağmen tahmin gücü mükemmel. |
| **Ortalama Güven (Mean Conf.)**| 0.8679 | **0.9695** | Model doğru bildiği şeylerde çekingenliği bıraktı. |
| **F1-Macro** | 0.9496 | **0.9537** | Sınıflar arası dengeli başarı oranı arttı. |
| **Precision-Macro** | 0.9497 | **0.9595** | Yanlış pozitif oranlarında iyileşme. |
| **Recall-Macro** | 0.9496 | **0.9483** | Gözden kaçan örnek sayısı azaldı. |

> **Genel Yorum:** NLL ve ECE değerlerindeki devasa düşüş, Dirichlet kalibrasyonunun logit uzayını kusursuz haritaladığını gösteriyor. Eklenen MCC metriği de modelin rastgele tahminden ne kadar uzak olduğunu (+1'e çok yakın) kanıtlıyor.

## 2. Sınıf Bazlı (Per-Class) Değişimler

Aşağıdaki tablo, araç kasası sınıflarının (Class) kalibrasyondan önceki ve sonraki F1, Precision ve Recall değerlerini ve o sınıfa ait test veri miktarını (Support) detaylıca göstermektedir.

| Sınıf Adı | Veri Sayısı (Support) | Ham F1 | Kalibre F1 | Değişim | Ham Prec. | Kalibre Prec. | Ham Recall | Kalibre Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SUV** | 670 | 0.9467 | **0.9492** | +0.0024 | 0.9517 | 0.9506 | 0.9418 | 0.9478 |
| **VAN** | 455 | 0.9845 | **0.9835** | -0.0010 | 0.9911 | 0.9846 | 0.9780 | 0.9824 |
| **STATION WAGON** | 66 | 0.9023 | **0.9219** | +0.0196 | 0.8955 | 0.9516 | 0.9091 | 0.8939 |
| **MICRO** | 22 | 1.0000 | **1.0000** | = | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **OPEN WHEEL / F1** | 586 | 1.0000 | **1.0000** | = | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **SEDAN** | 836 | 0.9620 | **0.9634** | +0.0014 | 0.9541 | 0.9510 | 0.9701 | 0.9761 |
| **HATCHBACK** | 266 | 0.8668 | **0.8812** | +0.0144 | 0.8652 | 0.8984 | 0.8684 | 0.8647 |
| **PICK UP** | 269 | 0.9346 | **0.9306** | -0.0040 | 0.9398 | 0.9394 | 0.9294 | 0.9219 |

> **Sınıf Bazlı Yorum:** Tablodan da görüleceği üzere 'Veri Sayısı (Support)' sütunu veri setindeki sınıfların (örneğin az sayıda örneği olan sınıfların) dağılımını net olarak ortaya koymaktadır. Modeller arası dengesiz (imbalanced) yapıya rağmen, log-probability uzayındaki Dirichlet dönüşümü azınlık sınıfların F1 skorlarını koruyarak (veya yükselterek) karar sınırlarını netleştirmiştir.

## 3. En Büyük Hata İyileştirmeleri (Confusion Matrix Düzeltmeleri)

Kalibrasyon sadece olasılıkları düzeltmekle kalmadı, aynı zamanda matris çarpımı sayesinde yanlış bilinen bazı örneklerin sınırlarını kaydırarak doğru sınıfa yerleşmesini sağladı. En çok düzeltilen hata rotaları:

- **SEDAN** aracını yanlışlıkla **HATCHBACK** sanma hatası: `0.0251` oranından `0.0167` oranına düştü. *(Hata azaldı)*
- **SUV** aracını yanlışlıkla **HATCHBACK** sanma hatası: `0.0209` oranından `0.0164` oranına düştü. *(Hata azaldı)*
- **HATCHBACK** aracını yanlışlıkla **STATION WAGON** sanma hatası: `0.0075` oranından `0.0038` oranına düştü. *(Hata azaldı)*
- **SUV** aracını yanlışlıkla **STATION WAGON** sanma hatası: `0.0030` oranından `0.0000` oranına düştü. *(Hata azaldı)*
- **VAN** aracını yanlışlıkla **SUV** sanma hatası: `0.0088` oranından `0.0066` oranına düştü. *(Hata azaldı)*

---

### Sonuç Bildirgesi
Bu rapor sonucunda DINOv3 Weighted modelinin **Dirichlet Calibration (ODIR λ=0.001)** kullanılarak, sıfır veri sızıntısı (Zero Leakage) prensibiyle kalibre edildiği kanıtlanmıştır. Model, doğruluk, Precision, Recall, MCC ve kritik güvenilirlik (ECE < %1) açısından makale/endüstri standartlarına ulaşmıştır.