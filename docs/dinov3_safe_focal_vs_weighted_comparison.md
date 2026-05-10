# DINOv3 Safe Augmentation Ablation — Focal Loss vs Class-Weighted Loss

**Tarih:** 2026-05-10  
**Amaç:** Aynı DINOv3 ViT-S/16 backbone’u ve aynı safe augmentation hattı altında, yalnızca loss seçiminin etkisini karşılaştırmak.  
**Kısa karar:** Bu iki aday arasında **class-weighted loss** daha iyi genel sonuç verdi; focal loss ise yakın bir baseline olarak kaldı.

> Not: Bu karşılaştırma “loss odaklı ablation” olarak okunmalıdır. İki run arasında küçük bir confounder daha var: `batch_size` focal run’da 64, weighted run’da 128. Yani sonuçlar çok güçlü bir karşılaştırma olsa da, tam anlamıyla tek değişkenli deney değildir.

---

## Karşılaştırılan iki run

| Run | Klasör | Checkpoint | Loss tipi | Batch size | Önemli not |
|---|---|---|---|---:|---|
| Focal | `checkpoints/dinov3_safe_focal_aug_20260510_111158` | `best-32-0.9537.ckpt` | Focal Loss + class weights | 64 | Daha dengeli öğrenme bekleniyordu |
| Weighted | `checkpoints/autolens-ai_checkpoints_dinov3_safe_weighted_aug_20260510_131319` | `best-29-0.9491.ckpt` | Class-weighted CE | 128 | Daha iyi genel test sonucu verdi |

Her iki modelde de:
- aynı backbone: `vit_small_patch16_dinov3`
- aynı 8 sınıf
- aynı safe augmentation policy
- aynı internal test split (`3170` örnek)

---

## Grafikler

### Eğitim eğrileri

| Focal — Loss | Weighted — Loss |
|---|---|
| ![Focal training loss](../checkpoints/dinov3_safe_focal_aug_20260510_111158/training_loss.png) | ![Weighted training loss](../checkpoints/autolens-ai_checkpoints_dinov3_safe_weighted_aug_20260510_131319/training_loss.png) |
| ![Focal training accuracy](../checkpoints/dinov3_safe_focal_aug_20260510_111158/training_accuracy.png) | ![Weighted training accuracy](../checkpoints/autolens-ai_checkpoints_dinov3_safe_weighted_aug_20260510_131319/training_accuracy.png) |

### Normalize confusion matrix

| Focal | Weighted |
|---|---|
| ![Focal confusion matrix](../checkpoints/dinov3_safe_focal_aug_20260510_111158/confusion_matrix.png) | ![Weighted confusion matrix](../checkpoints/autolens-ai_checkpoints_dinov3_safe_weighted_aug_20260510_131319/confusion_matrix.png) |

> Önemli yorum: Loss grafiklerinin mutlak değerleri **doğrudan karşılaştırılamaz**, çünkü focal loss ve class-weighted cross entropy farklı amaç fonksiyonlarıdır. Buna rağmen iki grafik de aynı genel eğilimi gösteriyor: hızlı yakınsama, sonra plato.

---

## Ana sonuçlar

| Metrik | Focal | Weighted | Fark |
|---|---:|---:|---:|
| Accuracy | 0.9505 | **0.9584** | +0.0079 |
| Precision-macro | 0.9358 | **0.9561** | +0.0203 |
| Recall-macro | 0.9269 | **0.9456** | +0.0187 |
| F1-macro | 0.9302 | **0.9507** | +0.0205 |
| Precision-weighted | 0.9512 | **0.9586** | +0.0074 |
| Recall-weighted | 0.9505 | **0.9584** | +0.0079 |
| F1-weighted | 0.9502 | **0.9584** | +0.0082 |

### Kısa yorum

- **Weighted model**, tüm ana test metriklerinde daha iyi.
- Focal model kötü değil; ancak bu veri ve bu pipeline için **class-weighted CE daha güçlü bir seçim** olmuş.
- En büyük fark **macro recall** ve **macro F1** tarafında geliyor; bu da weighted modelin zor sınıfları daha iyi tuttuğunu gösteriyor.

---

## Class bazlı sonuçlar

### Focal

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUV | 0.9171 | 0.9582 | 0.9372 | 670 |
| VAN | 1.0000 | 0.9670 | 0.9832 | 455 |
| STATION WAGON | 0.8133 | 0.9242 | 0.8652 | 66 |
| MICRO | 0.9524 | 0.9091 | 0.9302 | 22 |
| OPEN WHEEL / F1 | 0.9983 | 0.9949 | 0.9966 | 586 |
| SEDAN | 0.9423 | 0.9761 | 0.9589 | 836 |
| HATCHBACK | 0.9156 | 0.8158 | 0.8628 | 266 |
| PICK UP | 0.9474 | 0.8699 | 0.9070 | 269 |

### Weighted

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUV | 0.9576 | 0.9433 | 0.9504 | 670 |
| VAN | 0.9933 | 0.9780 | 0.9856 | 455 |
| STATION WAGON | 0.9524 | 0.9091 | 0.9302 | 66 |
| MICRO | 1.0000 | 0.9545 | 0.9767 | 22 |
| OPEN WHEEL / F1 | 1.0000 | 0.9983 | 0.9991 | 586 |
| SEDAN | 0.9496 | 0.9689 | 0.9591 | 836 |
| HATCHBACK | 0.8667 | 0.8797 | 0.8731 | 266 |
| PICK UP | 0.9296 | 0.9331 | 0.9314 | 269 |

### F1 farkı: Weighted - Focal

| Class | Focal F1 | Weighted F1 | Delta |
|---|---:|---:|---:|
| SUV | 0.9372 | **0.9504** | +0.0132 |
| VAN | 0.9832 | **0.9856** | +0.0024 |
| STATION WAGON | 0.8652 | **0.9302** | +0.0650 |
| MICRO | 0.9302 | **0.9767** | +0.0465 |
| OPEN WHEEL / F1 | 0.9966 | **0.9991** | +0.0025 |
| SEDAN | 0.9589 | **0.9591** | +0.0002 |
| HATCHBACK | 0.8628 | **0.8731** | +0.0103 |
| PICK UP | 0.9070 | **0.9314** | +0.0244 |

**Özet:** Weighted model **her sınıfta** F1 tarafında daha iyi. En büyük sıçrama `STATION WAGON`, `MICRO` ve `PICK UP` sınıflarında.

---

## Confusion matrix okuması

### Focal modelde görülen başlıca hata kümeleri

- `HATCHBACK → SEDAN` karışması belirgin: hataların önemli kısmı burada.
- `PICK UP → SUV` karışması hâlâ güçlü.
- `STATION WAGON → SEDAN` ve kısmen `SUV` yönünde yanlış sınıflamalar var.
- `MICRO` sınıfı az örnekli olduğu için `SEDAN` ve `HATCHBACK` tarafına kayabiliyor.

### Weighted modelde görülen başlıca hata kümeleri

- `HATCHBACK → SEDAN` hatası devam ediyor ama oranı daha düşük.
- `PICK UP → SUV` karışması azaltılmış.
- `STATION WAGON` hâlâ zor sınıf; ancak focal’a göre daha iyi.
- `OPEN WHEEL / F1` neredeyse çözüldü; bu sınıfın ayrımı çok güçlü.

### Grafik üzerinden kısa yorum

- Her iki modelde de öğrenme hızlı başlıyor ve ilk ~10-15 epoch içinde büyük ölçüde plato yapıyor.
- Validation accuracy iki modelde de train accuracy’nin gerisinde kalıyor, ama fark aşırı değil; yani ciddi overfitting yok.
- Weighted modelin validation accuracy’si daha yüksek ve confusion matrix’i daha temiz.

---

## Hangi sınıflar “sorunsuz” sayılabilir?

Bu projede “sorunsuz” demek, modelin o sınıfta hem yüksek F1 vermesi hem de confusion matrix’te az karışması demek.

### Büyük ölçüde sağlam sınıflar

- **OPEN WHEEL / F1**: Neredeyse kusursuz.
- **VAN**: İki modelde de güçlü.
- **SEDAN**: Güçlü ve istikrarlı.

### Orta riskli sınıflar

- **SUV**: Genel olarak iyi; fakat `PICK UP` ile karışma izleri var.
- **PICK UP**: Weighted ile belirgin iyileşmiş ama SUV yönünde hata tamamen bitmemiş.

### Geliştirilmesi gereken sınıflar

- **STATION WAGON**: En kırılgan sınıflardan biri.
- **HATCHBACK**: Özellikle `SEDAN` ile karışıyor.
- **MICRO**: Veri azlığı nedeniyle küçük örnek setinde oynak davranıyor.

---

## Daha da geliştirmek için ne yapılabilir?

Bu iki modelden sonra artık kazanımın ana kaynağı büyük olasılıkla **loss değişimi değil, veri ve hata odaklı iyileştirme** olur.

### 1) Zor sınıflara daha fazla veri eklemek

Öncelik sırası:
1. `STATION WAGON`
2. `HATCHBACK`
3. `PICK UP`
4. `MICRO`

Bu sınıflarda daha fazla ve daha temiz örnek, modelin genel F1’ini loss değiştirmekten daha fazla yükseltebilir.

### 2) Sınıf bazlı augmentation

Özellikle silueti bozmayan augmentasyonlar:
- hafif crop / resize
- küçük perspektif dönüşümleri
- renk/aydınlık oynatmaları

`HATCHBACK` ve `STATION WAGON` için çok agresif augmentasyon yerine şekli koruyan augmentasyonlar daha mantıklı.

### 3) Deneyi daha “temiz” hale getirmek

Bu karşılaştırmada batch size farklıydı. Sonraki iterasyonda:
- aynı batch size
- aynı seed
- aynı early stopping ayarı

ile focal vs weighted yeniden koşulursa, loss etkisi daha net izole edilir.

### 4) Focal loss’u tekrar ince ayarlamak

Focal loss burada kötü değil, ama bu kurulumda weighted kadar iyi sonuç vermedi.

Eğer tekrar denenecekse:
- `gamma` küçültülebilir
- `alpha` stratejisi sadeleştirilebilir
- `class_weights` + focal birlikte kullanımı yeniden kalibre edilebilir

### 5) Zor sınıflara odaklı hata analizi

En çok karışan sınıf çiftleri:
- `HATCHBACK ↔ SEDAN`
- `PICK UP ↔ SUV`
- `STATION WAGON ↔ SEDAN`

Bu üç çift için yanlış sınıflanan örnekler tek tek incelenirse, dataset etiketleme kalitesi veya augmentation kaynaklı hatalar da yakalanabilir.

### 6) Şu an için en mantıklı karar

Eğer hedef **final teslim** ise, bu iki run arasında **weighted model** daha mantıklı seçim.

- Daha iyi overall metrics
- Daha iyi class-wise F1
- Daha temiz confusion matrix
- Focal’a göre daha güvenli final aday

Dolayısıyla mevcut aşamada büyük mimari değişim şart değil; asıl iş **hata sınıflarını azaltmak** ve **veri kalitesini artırmak**.

---

## Son karar

Bu iki aday arasında:

1. **Weighted loss modeli** daha iyi genel performans verdi.
2. Focal loss modeli kötü değil, ama bu veri setinde weighted kadar iyi değil.
3. Her iki model de güçlü sonuçlar üretiyor; fakat güvenli final seçim olarak **weighted** daha doğru.
4. Eğer daha fazla geliştirme yapılacaksa, bunun ana yolu artık loss değiştirmek değil, **sınıf bazlı veri/hata iyileştirmesi** olmalı.

**Kısa final cümle:**  
Bu ablation’da `class-weighted CE + safe augmentation + DINOv3` kombinasyonu, `focal loss + safe augmentation + DINOv3` kombinasyonunu geçti; dolayısıyla final çizgi için weighted model tercih edilmeli.
