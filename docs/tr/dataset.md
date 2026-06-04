# AutoLens AI — Veri Seti

AutoLens AI veri seti, araç gövde tipi sınıflandırması için farklı açık kaynaklardan toplanıp temizlenmiş özel bir veri setidir. Amaç, 8 hedef sınıf için dengeli, tekrarları azaltılmış ve etiketleri güvenilir bir eğitim/evaluation seti oluşturmaktır.

## Hedef Sınıflar

| Sınıf | Açıklama |
|---|---|
| SUV | Sport Utility Vehicle |
| VAN | Van / minibüs |
| STATION WAGON | Station wagon / estate |
| MICRO | Microcar / küçük şehir aracı |
| OPEN WHEEL / F1 | Formula / açık tekerlekli araç |
| SEDAN | Sedan |
| HATCHBACK | Hatchback |
| PICK UP | Pickup truck |

## Final Dağılım

| Sınıf | Train | Val | Internal Test | Toplam |
|---|---:|---:|---:|---:|
| SUV | 5,351 | 669 | 670 | 6,689 |
| VAN | 3,631 | 453 | 455 | 4,539 |
| STATION WAGON | 521 | 64 | 66 | 651 |
| MICRO | 165 | 21 | 22 | 206 |
| OPEN WHEEL / F1 | 4,677 | 583 | 586 | 5,846 |
| SEDAN | 6,681 | 834 | 836 | 8,351 |
| HATCHBACK | 2,121 | 264 | 266 | 2,651 |
| PICK UP | 2,143 | 265 | 269 | 2,679 |
| **Toplam** | **25,285** | **3,157** | **3,170** | **31,612** |

## Kurasyon Politikası

- Güvenli olmayan etiket dönüşümleri yapılmadı.
- `City Car` otomatik MICRO sayılmadı.
- `Truck` otomatik PICK UP sayılmadı; yalnızca gerçek pickup örnekleri alındı.
- MICRO sınıfı onaylı model whitelist'i ile oluşturuldu.
- Bozuk/okunamayan görseller çıkarıldı.
- Near-duplicate tespiti için aHash64 perceptual hash kullanıldı.
- Leakage riskini azaltmak için bazı kaynaklar dışlandı.

## Split Stratejisi

Veri seti stratified 80/10/10 oranıyla ayrıldı:

- Train: eğitim
- Validation: early stopping ve kalibrasyon fit'i
- Internal Test: final model seçimi ve rapor metrikleri

Internal test verisi eğitim ve kalibrasyon sırasında kullanılmadı.

## Yeniden Üretilebilirlik

Ham görseller lisans nedenleriyle GitHub'a eklenmez. Kaynak katalogları ve manifest dosyaları `artifacts/dataset/` altında tutulur. Veri setini yeniden üretmek için kaynaklar indirilip curation scriptleri çalıştırılmalıdır.