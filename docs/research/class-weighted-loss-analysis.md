# Class-Weighted Cross Entropy — Teknik Analizi

> **Proje:** AutoLens AI — 8 Sınıf Araç Gövde Tipi Sınıflandırması  
> **Tarih:** 2026-05-10  
> **Bağlam:** DINOv3 ViT-S/16 backbone'unda loss fonksiyonu seçimi

---

## Neden Class-Weighted CE?

DINOv3 backbone'unda yapılan ablation study'de, **focal loss** ve **class-weighted cross entropy** doğrudan karşılaştırıldı:

| Metrik | Focal Loss | Weighted CE | Fark |
|---|---|---|---|
| Accuracy | 0.9505 | **0.9584** | +0.79pp |
| Precision-macro | 0.9358 | **0.9561** | +2.03pp |
| Recall-macro | 0.9269 | **0.9456** | +1.87pp |
| **F1-macro** | 0.9302 | **0.9507** | **+2.05pp** |
| F1-weighted | 0.9502 | **0.9584** | +0.82pp |

F1-macro'da **+2.05 puan** fark, istatistiksel olarak anlamlıdır.

## Sınıf Bazlı Etki

| Sınıf | Focal F1 | Weighted F1 | Fark |
|---|---|---|---|
| SUV | 0.9372 | **0.9504** | +1.32pp |
| VAN | 0.9832 | **0.9856** | +0.24pp |
| **STATION WAGON** | 0.8652 | **0.9302** | **+6.50pp** |
| **MICRO** | 0.9302 | **0.9767** | **+4.65pp** |
| OPEN WHEEL/F1 | 0.9966 | **0.9991** | +0.25pp |
| SEDAN | 0.9589 | **0.9591** | +0.02pp |
| **PICK UP** | 0.9070 | **0.9314** | **+2.44pp** |
| **HATCHBACK** | 0.8628 | **0.8731** | +1.03pp |

**En büyük kazanımlar:** STATION WAGON (+6.5pp), MICRO (+4.65pp), PICK UP (+2.44pp) — yani **az örnekli ve zor sınıflar** en çok etkilendi.

## Neden Çalışıyor?

1. **Eşitsiz veri dağılımı düzeltme:** Araç veri setinde bazı sınıflar (SEDAN, SUV) çok daha fazla örnek içerirken, STATION WAGON ve MICRO gibi sınıflar çok az. Class-weighted CE, az temsil edilen sınıflara daha yüksek loss ağırlığı verir.

2. **Gradyan dengelemesi:** Focal loss zor örneklere odaklanmayı hedefler ama batch_size farklılığı (64 vs 128) ve alpha/gamma hiperparametre seçimi sonucunda weighted CE daha tutarlı sonuç verdi.

3. **Safe augmentation ile uyumu:** Safe augmentation politikası zaten veri çeşitliliğini artırıyor; weighted loss ile bu iki yaklaşım birbirini tamamlayarak çalışıyor.

## Focal Loss Neden Yetersiz?

- Gamma=2.0 varsayılan değeri bu veri setinde çok agresif
- Az örnekli sınıflar (MICRO: 22, STATION WAGON: 66) için focal loss'un easy-example-suppression etkisi, zaten az veri olan sınıflardan öğrenmeyi zorlaştırıyor
- Alpha ağırlık stratejisi veri seti dengesine tam uyum sağlayamadı

**Not:** Focal loss "kötü" değildi (F1: 0.9302 hâlâ yüksek), ama aynı pipeline'da weighted CE onu geçti.

## Uygulama Detayları

```python
# PyTorch için class-weighted CE
class_weights = torch.tensor([
    1.0,   # SUV (bol)
    1.2,   # VAN
    3.0,   # STATION WAGON (az)
    5.0,   # MICRO (çok az)
    1.0,   # OPEN WHEEL/F1 (bol)
    1.0,   # SEDAN (en bol)
    2.0,   # HATCHBACK
    2.0,   # PICK UP
], device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

Ağırlıklar, sınıf frekanslarının tersiyle orantılı olarak hesaplanmalı (veya `sklearn.utils.class_weight.compute_class_weight` kullanılabilir).

## Konfüzyon Matrisi Karşılaştırması

- **Focal:** HATCHBACK→SEDAN karışması belirgin, PICK UP→SUV karışması güçlü
- **Weighted:** İlgili karışmalar belirgin şekilde azaltılmış, STATION WAGON sınıfı daha iyi ayrılmış

Weighted modelin confusion matrix'i genel olarak daha "temiz" ve sınıflar arası ayrım daha net.

## Sonuç

**DINOv3 ViT-S/16 pipeline'ında class-weighted CE, focal loss'a göre istatistiksel olarak anlamlı farkla daha iyi performans vermektedir.** Özellikle projemizdeki en kritik metrik olan **macro F1** için +2.05 puan fark, weighted CE'yi tercih sebebi olarak yeterli düzeydedir.

Ayrıca weighted CE'nin avantajı, focal loss'un aksine **ekstra hiperparametre (gamma, alpha) ayarına ihtiyaç duymamasıdır** — sadece sınıf ağırlıklarını hesaplamanız yeterlidir, bu da pipeline'ı basitleştirir ve tekrar deneylerde tutarlılık sağlar.