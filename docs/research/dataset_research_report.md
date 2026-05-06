# AutoLens AI Dataset Araştırma ve Birleştirme Raporu

**Proje:** AutoLens AI — Yazlab 2 Proje 3  
**Amaç:** 8 sınıflı araba gövde tipi sınıflandırması için açık kaynak görsellerden dengeli, genelleme kabiliyeti yüksek ve raporda kaynakları izlenebilir bir custom dataset oluşturmak.  
**Hedef sınıflar:** `SUV`, `VAN`, `STATION_WAGON`, `MICRO`, `OPEN_WHEEL`, `SEDAN`, `HATCHBACK`, `PICK_UP`  
**Ana metrik önceliği:** F1-score; bu nedenle sınıf dengesi ve az temsil edilen sınıfların kalitesi Accuracy'den daha kritiktir.

## 1. Kullanılan araştırma kaynakları

Bu rapor iki yerel araştırma dokümanını birleştirerek hazırlanmıştır:

- `.planning/research/gemini_Car Image Datasets For Classification .md`
- `.planning/research/qwen_From_Scraped_Images_to_Labeled_Models_ A_Strategic_Guide_to_Compiling_a_Custom_Vehicle_Body_Type_Dataset.md`

Ek olarak, araştırma notlarındaki kritik dataset boyutları güncel web kaynaklarıyla kontrol edilmiştir. Kaggle sayfaları zaman zaman dinamik/oturum kaynaklı hata döndürebildiği için bazı Kaggle adaylarında araştırma notu + arama sonucu birlikte değerlendirilmiştir.

Ayrıca proje ana ister dokümanı olan `docs/Yazlab 2- Proje 3.md` içinde iki dataset açıkça **referans ve ilham kaynağı** olarak verilmiştir:

- **Referans Veri Seti 1 — Cars Body Type Cropped:** <https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped>
- **Referans Veri Seti 2 — Stanford Car Body Type Data:** <https://www.kaggle.com/datasets/mayurmahurkar/stanford-car-body-type-data>

Bu iki kaynak raporda bu nedenle ayrı öncelikli değerlendirilmiştir. Ancak ödev metni bu veri setlerini doğrudan kullanmayı zorunlu kılmıyor; farklı açık kaynaklardan toplanan görsellerle custom dataset oluşturulmasını ve kullanılan kaynakların raporda belirtilmesini istiyor.

## 2. Projeye göre veri seti seçme mantığı

Bu projede en büyük risk yalnızca “çok görsel toplamak” değil; sınıfların görsel alanlarının birbirine karışmasıdır. Özellikle:

- `SEDAN` ve `HATCHBACK` ön açıdan çok benzer görünebilir.
- `HATCHBACK`, `STATION_WAGON` ve `MICRO` etiketleri bazı veri setlerinde aynı üst kategoriye sıkışabilir.
- `PICK_UP`, birçok veri setinde “truck” olarak geçer; ağır kamyonlarla karışmamalıdır.
- `OPEN_WHEEL`, normal trafik verisinde bulunmaz; F1 / formula / single-seater kaynaklarından ayrı toplanmalıdır.
- `VAN`, `MUV`, `MPV`, minibüs ve tempo-traveller gibi sınıflarla karışabilir; proje için binek/orta boy van görüntüleri tercih edilmelidir.

Bu yüzden önerilen strateji: **datasetleri ham boyutları oranında birleştirmek yerine, sınıf başına eşit hedef kota ile örneklemek**. Büyük datasetler çeşitlilik sağlar ama doğrudan eklenirse model büyük kaynakların arka plan/çekim tarzına ezber yapabilir.

## 3. Dataset adayları ve proje uygunluğu

| Dataset | Kaynak | Yaklaşık boyut | Projedeki sınıf karşılığı | Label durumu / otomatik çıkarım | Uygunluk | Not |
|---|---:|---:|---|---|---|---|
| Car Body Types Images / Cars Body Type Cropped | Kaggle | ~7.000 görsel, 7 sınıf, sınıf başı ~1.000 | `SUV`, `VAN`, `SEDAN`, `HATCHBACK`, `PICK_UP` | **Hazır label var.** Klasör adları doğrudan label olarak okunabilir. Manuel tek tek etiketleme gerekmez; yalnız sınıf dışı klasörleri elemek gerekir. | **Çok yüksek** | **Ana ödev dokümanında Referans Veri Seti 1 olarak verilmiş.** Doğrudan gövde tipi klasörleri var. `Convertible`, `Coupe` dışarıda bırakılmalı; `Coupe` sedan'a otomatik eklenmemeli. |
| VehicleType-10 | Kaggle competition | 1.002 görsel, 10 sınıf | `SUV`, `VAN`, `SEDAN`, `HATCHBACK`, `PICK_UP`, `STATION_WAGON` | **Hazır label var.** Yarışma klasörü/CSV yapısından label alınabilir. Manuel label gerekmez; sadece hedef 8 sınıfa remap gerekir. | **Yüksek** | Küçük ama proje taksonomisine çok yakın. Station wagon için kritik başlangıç kaynağı. |
| Vehicle Images Dataset | Kaggle | 15.645 görsel, 7 sınıf | `SUV`, `VAN`, `SEDAN`, kısmen `MICRO`/`HATCHBACK` | **Kısmen hazır label var.** Klasör adları okunabilir; ancak `City Car` ve `Truck` proje sınıfları için doğrudan güvenli değildir. `City Car` içinden micro/hatchback filtreleme gerekir. | **Orta-yüksek** | `City Car` mikro, hatchback ve city car karışımıdır; doğrudan kullanılmamalı, filtrelenmeli. `Truck` pickup değildir. |
| Stanford Cars | Stanford/Kaggle/HF yansımaları | 16.185 görsel, 196 make-model-year sınıfı | Remap ile 6 sınıf | **Hazır fine-grained label var, body-type yok.** CSV/MAT/class-name dosyasından make-model-year alınır; body type için otomatik lookup/remap tablosu gerekir. Tek tek görsel label gerekmez ama remap tablosu elle hazırlanır. | **Orta-yüksek** | Kaliteli benchmark; gövde tipi etiketi yok. Sınıf adlarından/make-model tablosundan remap gerekir. |
| Stanford Car Body Type Data | Kaggle | Stanford türevi | Birden fazla gövde tipi | **Muhtemelen hazır body-type label var.** Ödevin referans verdiği Stanford türevi olduğundan klasör/CSV label beklenir; indirme sonrası kolon/klasör isimleri doğrulanmalı. | **Yüksek aday** | **Ana ödev dokümanında Referans Veri Seti 2 olarak verilmiş.** İndirildikten sonra sınıf dağılımı ve label formatı doğrulanmalı. |
| CompCars | Akademik / CUHK / Kaggle mirror | 136.726 full-car web görseli + 50.000 surveillance | 12 tip hiyerarşisinden remap | **Metadata label var.** Model attribute dosyalarında car type bilgisi bulunur; CSV/metadata parse edilerek body type çıkarılabilir. Manuel tek tek label gerekmez, fakat 12 tipten proje 8 sınıfına remap gerekir. | **Yüksek ama işçilikli** | Gövde tipi attribute'u avantaj. Lisans/erişim ve remap pipeline'ı kontrol edilmeli. |
| VMMRdb | GitHub/Kaggle mirror | 291.752 görsel, 9.170 sınıf | Make-model-year remap ile özellikle `MICRO`, `STATION_WAGON`, `PICK_UP` | **Hazır make-model-year label var, body-type yok.** Klasör/metadata label'ı otomatik okunabilir; body type için model lookup tablosu gerekir. Micro/station/pickup için iyi ama remap kalitesi kritik. | **Yüksek ama gürültülü** | Büyük ve çeşitli; sınıf isimleri model bazlı olduğu için lookup tablosu şart. |
| UVH-26 | Hugging Face | 26.646 1080p trafik karesi, 14 araç sınıfı | `HATCHBACK`, `SEDAN`, `SUV`, `VAN` crop | **Hazır object-detection label var.** COCO JSON içindeki `category_id` ve `bbox` ile araç crop + label otomatik çıkarılabilir. Tek tek label gerekmez; crop pipeline gerekir. | **Orta-yüksek** | Sınıflandırma değil object detection datasetidir. Bounding box crop çıkarılırsa gerçek trafik genellemesi katar. |
| BoxCars116k | Akademik / OpenDataLab mirror | ~116.000 araç görseli | Model/type remap ile 5-6 sınıf | **Metadata var ama doğrudan proje label'ı değil.** Araç/model annotationlarından body type çıkarmak için lookup/remap gerekir. Manuel tek tek label yerine metadata tabanlı remap yapılabilir. | **Orta** | Surveillance domain için değerli; doğrudan 8 gövde tipi klasörü değildir. |
| F1 Image Classification Updated | Kaggle | Araştırma notuna göre ~5.300+ | `OPEN_WHEEL` | **Hazır klasör label'ı var ama hedef için tek sınıfa indirgenir.** Constructor/takım klasörleri varsa hepsi `OPEN_WHEEL` olarak merge edilir. Manuel label gerekmez; kapalı tekerlekli görsel sızıntısı kontrol edilir. | **Çok yüksek aday** | Open-wheel sınıfı için ana kaynak olabilir; indirme sonrası gerçek dosya sayısı doğrulanmalı. |
| Formula One Cars | Kaggle | Araştırma notuna göre ~2.800 | `OPEN_WHEEL` | **Hazır/yarı hazır label var.** F1 araç klasörleri veya metadata tek hedef label'a (`OPEN_WHEEL`) dönüştürülebilir. Manuel tek tek label gerekmez; sadece F1 dışı görseller temizlenir. | **Yüksek aday** | F1 sınıfını büyütmek için kullanılabilir; takım/constructor klasörleri tek `OPEN_WHEEL` altında birleştirilmeli. |
| Racecar V4 / Roboflow yarış projeleri | Roboflow | Araştırma notuna göre ~1.140 | `OPEN_WHEEL` ve yarış araçları | **Hazır detection label olabilir ama hedef label karışık.** Roboflow annotationları parse edilebilir; ancak GT/NASCAR/rally sınıfları ayrılmalı. Open-wheel olmayanları filtrelemek için manuel audit veya model destekli kontrol gerekir. | **Orta** | GT/NASCAR/rally karışabilir; sadece açık tekerlekli araçlar manuel/otomatik filtrelenmeli. |
| Car Model Variants and Images Dataset | Kaggle | ~193.000 görsel, body type metadata | Remap ile birçok sınıf | **CSV/metadata ile güçlü otomatik label çıkarımı var.** `body type` kolonu doğrudan parse edilip proje sınıflarına remap edilebilir. Tek tek label gerektirmeyen en iyi ek adaylardan biridir. | **İyi ek aday** | Metadata'da body type olduğu için ikinci tur zenginleştirme için güçlü adaydır; lisans ve kalite kontrol gerekli. |

## 3.1 Label çıkarma açısından önceliklendirme

Datasetleri indirme sırası yalnız görsel sayısına göre değil, **label otomasyon kolaylığına** göre de seçilmelidir:

1. **En hızlı kullanılacaklar:** Car Body Types Images, VehicleType-10, F1 datasetleri. Bunlarda klasör/CSV label hazırdır; sadece proje sınıf isimlerine remap yapılır.
2. **CSV/metadata ile güçlü adaylar:** CompCars ve Car Model Variants. Body type bilgisi metadata içinde olduğu için elle tek tek etiketleme gerektirmez; parse + remap scripti gerekir.
3. **Make-model lookup gerektirenler:** Stanford Cars, VMMRdb, BoxCars. Görsel label'ı vardır ama gövde tipi yoktur; make/model/year -> body type lookup tablosu hazırlanmalıdır. Bu iş tek tek görsel etiketlemekten daha hızlıdır.
4. **Detection crop gerektirenler:** UVH-26 ve Roboflow. Label JSON/annotation dosyasında vardır; sınıflandırma dataseti üretmek için bounding box crop çıkarma pipeline'ı gerekir.
5. **Manuel audit gerektiren riskli sınıflar:** `MICRO`, `PICK_UP`, `OPEN_WHEEL`. Bu sınıflarda otomatik label olsa bile son örneklerin bir kısmı hızlı görsel kontrolden geçirilmelidir.

Pratik sonuç: İlk dataset sürümünde elle tek tek label yazmak yerine, **klasör adı, CSV kolonu, COCO JSON category_id veya make-model lookup** üzerinden otomatik label üretmek mümkündür. Manuel iş yalnızca gürültülü sınıfların kalite kontrolüne ayrılmalıdır.

## 4. Ham boyuta göre değil, dengeli kota ile birleştirme

Aşağıdaki ham boyutlar gösteriyor ki datasetleri doğrudan oranlarsak VMMRdb, CompCars ve BoxCars gibi büyük kaynaklar küçük ama temiz body-type datasetlerini ezer:

| Kaynak grubu | Yaklaşık ham hacim | Doğrudan eklenirse risk |
|---|---:|---|
| VMMRdb | 291.752 | Gürültülü make-model etiketleri ve web arka planı baskın olur. |
| CompCars | 136.726 + 50.000 | Büyük hacim iyi ama lisans/remap ve domain dengesi yönetilmeli. |
| BoxCars116k | ~116.000 | Surveillance domain baskınlaşır; bazı sınıflar eksik kalır. |
| UVH-26 | 26.646 frame | Detection crop gerektirir; Hindistan trafik domain'i tek başına baskın olmamalı. |
| Stanford Cars | 16.185 | Kaliteli ama showroom/benchmark bias yaratabilir. |
| Vehicle Images Dataset | 15.645 | `City Car` ve `Truck` sınıfları proje için gürültülü. |
| Car Body Types Images | ~7.000 | Temiz ama 5 hedef sınıfı kapsar, station/micro/open-wheel eksik. |
| VehicleType-10 | 1.002 | Çok uyumlu ama hacmi küçük. |
| F1 adayları | ~2.800-5.300+ | Open-wheel için şart; arka plan çoğunlukla pist olduğu için augmentasyon gerekir. |

**Sonuç:** Birleştirme algoritması “kaynak ne kadar büyükse o kadar çok al” olmamalı. Bunun yerine her hedef sınıf için aynı sayıda seçilmiş, temizlenmiş ve domain bakımından karışık görsel alınmalıdır.

## 5. Önerilen hedef dağılım

Nihai DINOv3 ana model hedefi için önerilen dağılım **sınıf başı yaklaşık 4.000 temiz görsel, toplam yaklaşık 32.000 temiz görsel** olmalıdır. Ham indirme 36.000–40.000 aday görsel tutulur; audit ve dedup sonrası sınıflar 3.5k–4.3k bandında dengelenir.

| Sınıf | Minimum kabul | Tercih edilen hedef | Neden |
|---|---:|---:|---|
| `SUV` | 3.700 | 4.000 | Kaynak bol; fazla kolay sınıf olmaması için diğerleriyle dengelenmeli. |
| `VAN` | 3.700 | 4.000 | Van/minibüs/MPV gürültüsü temizlenmeli. |
| `STATION_WAGON` | 3.500 | 4.000 | Az bulunan ve hatchback ile karışan sınıf; kalite kritik. |
| `MICRO` | 3.200 | 4.000 | En zor sınıflardan biri; model bazlı filtre gerekir. |
| `OPEN_WHEEL` | 3.500 | 4.000 | Ayrı domain; F1 görüntüleri pist bias'ı taşıyabilir. |
| `SEDAN` | 3.700 | 4.000 | Kaynak bol; hatchback/coupe karışımı temizlenmeli. |
| `HATCHBACK` | 3.700 | 4.000 | Sedan ve micro ile sınırı netleştirilmeli. |
| `PICK_UP` | 3.500 | 4.000 | Ağır/generic truck görüntüleri dışarıda bırakılmalı. |

## 6. Sınıf bazlı kaynak önerisi

### 6.1 `SUV`

Önerilen karışım:

- %40 Car Body Types Images / Cars Body Type Cropped
- %20 VehicleType-10
- %20 CompCars veya Stanford remap
- %20 UVH-26 / BoxCars gibi trafik-surveillance crop'ları

Not: SUV sınıfı bol olduğu için fazla kolay olur. Aşırı temiz katalog görselleri yerine farklı açı, ışık ve arka planlardan örnek seçilmeli.

### 6.2 `VAN`

Önerilen karışım:

- %35 Car Body Types Images
- %20 VehicleType-10
- %20 Vehicle Images Dataset `Van`
- %15 UVH-26 `Van` crop
- %10 Stanford/CompCars remap

Dikkat: Minibüs, otobüs, tempo-traveller, campervan ve MPV örnekleri ayrı kontrol edilmeli. Proje sınıfı “van” olduğu için aşırı büyük minibüs/otobüs dışarıda bırakılmalıdır.

### 6.3 `STATION_WAGON`

Önerilen karışım:

- %30 VehicleType-10 station wagon
- %25 Stanford Cars remap (`wagon`, `estate`, bazı model adları)
- %25 CompCars / VMMRdb model remap
- %20 BoxCars veya manuel doğrulanmış ek kaynak

Dikkat: Station wagon sınıfı hatchback'in uzun gövdeli versiyonu gibi görünebilir. Yan ve arka-çapraz açı oranı yüksek tutulmalıdır; yalnız ön görünüşler sınıfı öğretmekte yetersizdir.

### 6.4 `MICRO`

Önerilen karışım:

- %35 Vehicle Images Dataset içindeki `City Car` klasöründen manuel/otomatik filtrelenmiş gerçek micro car örnekleri
- %35 VMMRdb veya Car Model Variants üzerinden model bazlı seçim
- %20 Stanford/CompCars model remap
- %10 kontrollü web/Kaggle ek kaynak

Örnek güvenli model anahtarları: Smart ForTwo, Smart ForFour, Fiat 500, Toyota iQ, Renault Twizy, Mitsubishi i-MiEV, kei car modelleri.  
Dikkat: Her küçük hatchback micro değildir. Micro sınıfı için kısa dingil mesafesi, çok kısa ön/arka çıkıntı ve kompakt kabin aranmalıdır.

### 6.5 `OPEN_WHEEL`

Önerilen karışım:

- %50 F1 Image Classification Updated
- %30 Formula One Cars
- %20 Roboflow/yarış datasetlerinden yalnız açık tekerlekli crop'lar

Dikkat: GT, NASCAR, Le Mans, rally ve kapalı tekerlekli yarış otomobilleri bu sınıfa eklenmemelidir. F1 görselleri çoğunlukla pistte olduğu için augmentasyonda renk, crop, blur, viewpoint ve background çeşitlendirmesi yapılmalıdır; ama açık tekerlek ve kanat morfolojisini bozacak agresif crop'lardan kaçınılmalıdır.

### 6.6 `SEDAN`

Önerilen karışım:

- %35 Car Body Types Images
- %20 VehicleType-10
- %20 Stanford/CompCars remap
- %15 UVH-26 crop
- %10 VMMRdb/BoxCars

Dikkat: Coupe örnekleri sedan'a otomatik eklenmemeli. Proje örneği 4 kapılı klasik sedanı temsil ediyor; iki kapılı coupe'ler sınıf sınırını bulanıklaştırabilir.

### 6.7 `HATCHBACK`

Önerilen karışım:

- %35 Car Body Types Images
- %20 VehicleType-10
- %20 UVH-26 `Hatchback` crop
- %15 Stanford/CompCars remap
- %10 Vehicle Images Dataset `City Car` içinden hatchback olarak doğrulanan örnekler

Dikkat: Micro car ve station wagon örnekleri hatchback'e karışmamalı. Hatchback için arka kapak/truncated rear görünürlüğü olan yan/arka açı örnekleri özellikle seçilmeli.

### 6.8 `PICK_UP`

Önerilen karışım:

- %45 Car Body Types Images `Pick-Up`
- %25 VehicleType-10 pickup
- %20 Stanford/CompCars/VMMRdb model remap
- %10 manuel doğrulanmış ek kaynak

Dikkat: Generic `Truck`, `Big Truck`, kamyon, çekici, dump truck ve delivery truck görselleri pick-up değildir. Pick-up için açık kasa ve binek kabini aynı anda görünmelidir.


## 7. 30–40k hedef için nihai dataset kararı

Bu proje için nihai hedef **ham olarak 30–40k arası indirmek**, temizleme/audit sonrası ise **yaklaşık 32.000 dengeli görsel** elde etmektir. En pratik hedef **8 sınıf × 4.000 temiz görsel = 32.000 görsel** olmalıdır. İndirme sırasında bazı görseller duplicate, yanlış label, düşük kalite veya sınıf dışı çıkacağı için ham indirme kotası **36.000–40.000 aday görsel** tutulabilir.

### 7.1 Hedef sınıf dağılımı

| Sınıf | Temiz final hedef | Kabul aralığı | Not |
|---|---:|---:|---|
| `SUV` | 4.000 | 3.700–4.300 | Kaynak bol; fazla şişirilmemeli. |
| `VAN` | 4.000 | 3.700–4.300 | Van/minibüs/MPV ayrımı kontrol edilmeli. |
| `STATION_WAGON` | 4.000 | 3.500–4.200 | Zor sınıf; gerekirse 3.500 altına düşmemeli. |
| `MICRO` | 4.000 | 3.200–4.000 | En riskli label; kalite, sayıdan daha önemli. |
| `OPEN_WHEEL` | 4.000 | 3.500–4.200 | F1/open-wheel dışı yarış araçları atılmalı. |
| `SEDAN` | 4.000 | 3.700–4.300 | Coupe karışımı engellenmeli. |
| `HATCHBACK` | 4.000 | 3.700–4.300 | Micro ve station wagon karışımı engellenmeli. |
| `PICK_UP` | 4.000 | 3.500–4.200 | Generic truck değil, açık kasalı pickup aranmalı. |
| **Toplam** | **32.000** | **28.500–34.700 temiz** | Ham indirme 36–40k olabilir. |

Bu hedef, DINOv3 ana model yolu için 12k'ya göre daha sağlıklı bir eğitim seti sağlar; yine de dataset tek bir kaynaktan değil, sınıf başına birkaç kaynaktan dengeli karıştırılmalıdır.

### 7.2 Datasetlerden hangi label'lar alınacak?

| Kaynak | Alınacak label'lar | Alınmayacak / dikkat edilecek label'lar | Kullanım kararı |
|---|---|---|---|
| Car Body Types Images / Cars Body Type Cropped | `SUV`, `VAN`, `SEDAN`, `HATCHBACK`, `Pick-Up` → `PICK_UP` | `Coupe`, `Convertible` dışarıda. Coupe sedan'a otomatik eklenmez. | İlk indirilecek çekirdek kaynak. Hazır klasör label'ı güvenilir. |
| VehicleType-10 | `SUV`, `VAN`, `SEDAN`, `HATCHBACK`, `PICKUP`, `STATION_WAGON` | Hedef dışı kalan iki sınıf alınmaz. | Küçük ama çok temiz; özellikle `STATION_WAGON` için alınır. |
| Stanford Car Body Type Data | Klasör/CSV'de varsa 8 hedefe uyan body-type label'lar | İndirince gerçek dağılım doğrulanır. | Ana dokümanda referans olduğu için öncelikli kontrol edilir. |
| Vehicle Images Dataset | `SUV`, `VAN`, `SEDAN`; `City Car` sadece model/görsel filtresiyle `MICRO` veya `HATCHBACK` | `City Car` doğrudan tek label'a atanmaz. `Truck` generic ise `PICK_UP` yapılmaz. | Yardımcı kaynak; ambiguous label'lar audit gerektirir. |
| Car Model Variants and Images Dataset | CSV/metadata `body type` kolonundan `SUV`, `VAN`, `SEDAN`, `HATCHBACK`, `WAGON/ESTATE`, `PICKUP`, micro modeller | Belirsiz body type veya spor/coupe label'ları dışarıda. | 30–40k hedef için en güçlü tamamlayıcı kaynaklardan biri. |
| CompCars | Metadata type: `SUV`, `Sedan`, `Hatchback`, `Estate` → `STATION_WAGON`, `Pickup`, uygun `MPV/Minibus` → kontrollü `VAN` | `Sports`, `Crossover`, `Fastback`, `Hardtop` doğrudan alınmaz. | Büyük ve metadata'lı; sınıf açığını kapatmak için kullanılır. |
| Stanford Cars / VMMRdb / BoxCars | Make-model-year lookup ile net gövde tipi çıkanlar | Lookup sonucu belirsizse alınmaz. | Sınıf başı 4k hedefte eksik kalan sınıfları tamamlar. |
| UVH-26 | COCO/detection label crop: `sedan`, `hatchback`, `suv`, `van` benzeri net sınıflar | Full frame kullanılmaz; bbox crop şart. | Gerçek trafik domain'i için sınıf başına sınırlı quota ile alınır. |
| F1 Image Classification Updated / Formula One Cars | Tüm gerçek F1 / formula / single-seater görselleri → `OPEN_WHEEL` | GT, NASCAR, rally, kapalı tekerlekli racecar alınmaz. | `OPEN_WHEEL` ana kaynağı. |
| Roboflow racecar projeleri | Sadece annotation/görsel olarak açık tekerlekli olanlar | GT/NASCAR/rally karışımı yüksek; otomatik label'a kör güvenilmez. | F1 kaynakları yetmezse tamamlayıcı. |

### 7.3 Ambiguous label kuralları

- **`City Car` doğrudan `MICRO` değildir.** Bu label hem micro car hem küçük hatchback içerebilir. Bu yüzden `City Car` klasörü doğrudan tek sınıfa atanmayacak. Yalnız şu iki yolla alınacak:
  1. Dosya/metadata/model adı Smart ForTwo, Fiat 500, Toyota iQ, Renault Twizy, kei car vb. net micro model ise `MICRO`.
  2. Görsel audit ile gerçekten çok kısa dingil mesafeli micro car olduğu doğrulanırsa `MICRO`.
- **`Truck` doğrudan `PICK_UP` değildir.** Eğer dataset label'ı `Pickup`, `Pick-Up`, `Pickup Truck`, `Ute` ise alınabilir. Sadece `Truck` yazıyorsa şu kontrol gerekir:
  - Açık arka kasa + binek kabini varsa `PICK_UP`.
  - Kamyon, çekici, dump truck, box truck, delivery truck, heavy truck ise dışarıda.
- **`Crossover` doğrudan `SUV` yapılmamalı.** Küçük crossover örnekleri SUV/hatchback sınırını bozabilir. Yalnız görsel olarak SUV morfolojisi belirgin olanlar alınmalı veya bu label başlangıçta dışarıda bırakılmalı.
- **`MPV/Minivan/Minibus` doğrudan `VAN` değildir.** Proje örneğine yakın van/minivan alınabilir; otobüs/minibüs/çok büyük ticari araçlar dışarıda kalır.
- **`Coupe` sedan'a eklenmez.** Sedan sınıfı klasik 3-box, çoğunlukla 4 kapılı otomobillerden oluşturulmalı.

### 7.4 Önerilen 36–40k ham indirme planı

| Aşama | Kaynak grubu | Ham aday hedef | Beklenen temiz katkı | Amaç |
|---|---|---:|---:|---|
| 1 | Assignment referansları: Cars Body Type Cropped + Stanford Car Body Type Data | 8k–12k | 6k–9k | Temiz hazır label çekirdeği. |
| 2 | VehicleType-10 + kitrofimov/cbsc gibi küçük body-style kaynakları | 1k–2k | 800–1.500 | Station wagon ve küçük doğrulama katkısı. |
| 3 | Car Model Variants + CompCars metadata filtreleri | 12k–16k | 10k–14k | 4k/sınıf hedefine ana tamamlayıcı. |
| 4 | Stanford/VMMRdb/BoxCars make-model lookup | 8k–10k | 6k–8k | Micro, station wagon, pickup ve domain çeşitliliği. |
| 5 | F1/Open-wheel özel kaynakları | 5k–7k | 3.5k–4.5k | `OPEN_WHEEL` sınıfını doldurma. |
| 6 | UVH-26 crop / trafik domain'i | 2k–3k | 1.5k–2.5k | Gerçek yol/trafik genellemesi. |
| **Toplam** |  | **36k–50k aday** | **28k–40k temiz** | Nihai hedef 32k temiz. |

Eğer disk/indirme süresi nedeniyle 50k aday fazla gelirse, en güvenli pratik plan **yaklaşık 38k aday indirip 32k temiz final** çıkarmaktır.

### 7.5 Nihai karar

Bu proje için önerilen final dataset versiyonu:

- **Dataset adı:** `autolens_bodytype_v1_32k`
- **Final temiz hedef:** yaklaşık **32.000 görsel**
- **Sınıf hedefi:** yaklaşık **4.000 görsel / sınıf**
- **Ham indirme hedefi:** **36.000–40.000 aday görsel**; gerekirse açık tekerlekli/micro eksikleri için +5k ek aday
- **Split:** %70 train / %15 validation / %15 internal test
- **Ana kural:** Hazır label veya metadata varsa otomatik çıkar; ambiguous label'ları doğrudan sınıfa basma, önce filtrele/audit et.

Bu karar “balanced'a yakın” kalır ve sorun çıkarabilecek label karışımlarını engeller. Özellikle `City Car`, generic `Truck`, `Crossover`, `MPV/Minibus`, `Coupe` gibi label'lar doğrudan hedef sınıfa aktarılmamalıdır.

## 8. Önerilen veri bölme oranları

Sunumda görülecek test seti dışarıdan geleceği için kendi elimizdeki dataset üç parçaya ayrılmalıdır:

| Bölüm | Oran | Amaç |
|---|---:|---|
| Train | %70 | Model öğrenimi |
| Validation | %15 | Early stopping, hiperparametre seçimi, model karşılaştırma |
| Internal test | %15 | Final ön kontrol; training/validation kararlarına dahil edilmez |

Önemli: Aynı model/yıl/seri veya aynı görselin benzer kopyaları farklı split'lere düşmemelidir. Özellikle Stanford, VMMRdb ve web-scrape kaynaklarında duplicate/near-duplicate kontrolü yapılmalıdır. Aksi halde metrikler olduğundan yüksek çıkar.

## 9. Ön işleme ve kalite kontrol önerileri

1. **Kaynak manifestosu tut:** Her görsel için `source`, `original_label`, `mapped_label`, `license`, `split`, `hash`, `width`, `height` alanları kaydedilmeli.
2. **Etiket remap tablosu oluştur:** `sedan/saloon`, `station wagon/estate/wagon`, `pickup/ute`, `open-wheel/formula/single-seater` eşleştirmeleri açıkça yazılmalı.
3. **Truck filtrelemesi yap:** `PICK_UP` için açık kasa yoksa görsel dışarıda kalmalı.
4. **City car filtrelemesi yap:** `MICRO` için model adı veya görsel morfoloji doğrulanmalı.
5. **Bounding box crop kullan:** UVH-26 ve benzeri detection datasetlerinde tüm frame yerine araç crop'u kullanılmalı.
6. **Class-balanced sampling uygula:** Eğitimde `WeightedRandomSampler` veya class-balanced batch kullanılabilir; ancak veri setinin kendisi de dengeli tutulmalı.
7. **Domain dengesi kur:** Her sınıfta katalog, yol/trafik, farklı ülke/ışık/açı örnekleri bulunmalı.
8. **Augmentasyonu sınıfa göre dikkatli uygula:** Flip, color jitter, resize-crop, blur kullanılabilir; ancak open-wheel ve pickup gibi morfolojik ayırt edici parçaları kesen crop'lardan kaçınılmalı.

## 10. İlk uygulanacak pratik plan

1. **Ana klasör yapısını sabitle:**

```text
data/raw/<source_name>/...
data/interim/crops_or_filtered/...
data/processed/autolens_bodytype_v1/
  train/{SUV,VAN,STATION_WAGON,MICRO,OPEN_WHEEL,SEDAN,HATCHBACK,PICK_UP}/
  val/{...}/
  test/{...}/
```

2. **Önce temiz ve direkt kaynakları indir:**
   - Car Body Types Images / Cars Body Type Cropped
   - VehicleType-10
   - F1 Image Classification Updated veya Formula One Cars

3. **Eksik sınıfları tamamla:**
   - `STATION_WAGON`: Stanford/CompCars/VMMR remap
   - `MICRO`: Vehicle Images `City Car` + VMMR/Car Model Variants model filtresi
   - `OPEN_WHEEL`: F1 + Roboflow filtrelenmiş crop

4. **Sınıf başı yaklaşık 4.000 temiz görsel hedefle:**
   - Nihai hedef 8 sınıf × 4.000 = yaklaşık 32.000 temiz görseldir.
   - Ham indirme 36.000–40.000 aday görsel olabilir; audit sonrası kirli/duplicate örnekler atılır.
   - 5.000+ / sınıf ancak sınıf kalitesi bozulmadan ve diğer sınıflar da dengede kalıyorsa düşünülmelidir.

5. **Audit raporu üret:**
   - Sınıf dağılımı
   - Kaynak dağılımı
   - Çözünürlük dağılımı
   - Duplicate sayısı
   - Manuel reddedilen örnek sayısı

## 11. Nihai öneri

AutoLens AI için en doğru dataset stratejisi, **Kaggle body-type datasetlerini çekirdek**, **Stanford/CompCars/VMMRdb gibi büyük make-model datasetlerini tamamlayıcı**, **UVH-26/BoxCars gibi trafik datasetlerini genelleme artırıcı**, **F1 datasetlerini de open-wheel için özel kaynak** olarak kullanmaktır.

Önerilen final sürüm:

- Toplam: **yaklaşık 32.000 temiz görsel**
- Sınıf başı: **yaklaşık 4.000 görsel**
- Ham indirme: **36.000–40.000 aday görsel**
- Split: **%70 train / %15 val / %15 internal test**
- Kaynak yaklaşımı: Her sınıfta en az 2 farklı kaynak, mümkünse 3 farklı domain
- Öncelik: `MICRO`, `OPEN_WHEEL`, `STATION_WAGON`, `PICK_UP` sınıflarında label temizliği; `City Car` ve generic `Truck` gibi ambiguous label'ları doğrudan kullanmamak

Bu yapı, ödevin “kendi custom datasetini oluşturma”, “kaynakları raporda belirtme”, “sınıf dengesine dikkat etme” ve “görülmemiş sunum testlerinde genelleme” beklentileriyle uyumludur.

## 12. Kaynak linkleri

- Car Body Types Images Dataset: <https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped>
- VehicleType-10: <https://www.kaggle.com/competitions/vehicle-type-10-a-dataset-for-vehicle-body-type-classification>
- Vehicle Images Dataset: <https://www.kaggle.com/datasets/lyensoetanto/vehicle-images-dataset>
- Stanford Cars Dataset: <https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset>
- Stanford original Cars page: <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>
- CompCars / Papers With Code summary: <https://paperswithcode.com/dataset/compcars>
- CUHK CompCars homepage: <http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html>
- VMMRdb GitHub: <https://github.com/faezetta/VMMRdb>
- UVH-26 Hugging Face: <https://huggingface.co/datasets/iisc-aim/UVH-26>
- BoxCars116k project/mirror info: <https://hyper.ai/en/datasets/9213>
- F1 Image Classification Updated: <https://www.kaggle.com/datasets/loveymishra/f1-image-classification-updated>
- Formula One Cars: <https://www.kaggle.com/datasets/vesuvius13/formula-one-cars>
- Car Model Variants and Images Dataset: <https://www.kaggle.com/datasets/eimadevyni/car-model-variants-and-images-dataset>

## 13. Derinleştirilmiş label politikası ve tek tek indirme listesi

Bu bölüm, 30–40k ham aday görsel indirmek isteyen pratik çalışma için nihai indirme ve label karar listesidir. Amaç, indirdikten sonra her datasetin tamamını kör şekilde kullanmak değil; **güvenli label'ları otomatik almak, şüpheli label'ları filtrelemek, riskli label'ları dışarıda bırakmak**tır.

### 13.1 Genelleştirilmiş label güvenlik seviyeleri

| Seviye | Anlamı | Ne yapılacak? | Örnek |
|---|---|---|---|
| **A — Direkt güvenli** | Label proje sınıfıyla birebir veya güçlü sinonim. | Otomatik remap yapılabilir; sadece duplicate/bozuk görsel temizliği gerekir. | `Sedan` → `SEDAN`, `Hatchback` → `HATCHBACK`, `Pick-Up` → `PICK_UP` |
| **B — Koşullu güvenli** | Label hedefe yakın ama başka sınıfları da kapsayabilir. | Metadata, model adı veya hızlı görsel audit ile alınır. | `City Car`, `Crossover`, `MPV`, generic `Truck` |
| **C — Dışarıda bırak** | Label proje sınıfını bozacak kadar geniş veya hedef dışı. | İlk dataset sürümüne alınmaz. | `Coupe`, `Convertible`, `Big Truck`, `Bus`, `Sports`, `Racecar` generic |

### 13.2 Hedef sınıf bazlı remap sözlüğü

| Hedef sınıf | Direkt alınacak label/sinonim | Koşullu alınacak label | Direkt dışarıda bırakılacak label |
|---|---|---|---|
| `SEDAN` | `Sedan`, `Saloon` | Bazı `Passenger Car` kayıtları, ancak body/model net sedan ise | `Coupe`, `Convertible`, `Fastback`, `Sports` |
| `HATCHBACK` | `Hatchback`, `Compact Hatchback` | `City Car` yalnız görsel/model hatchback ise | `Micro` net olanlar, `Station Wagon`, `Coupe` |
| `STATION_WAGON` | `Station Wagon`, `Wagon`, `Estate` | `Touring`, `Avant`, `Variant`, `SportWagen` model adı netse | `Hatchback`, `SUV`, belirsiz `Crossover` |
| `SUV` | `SUV`, `Sport Utility Vehicle`, net `Off-road/4x4` | `Crossover` yalnız görsel SUV morfolojisi netse | Hatchback gibi duran compact crossover'lar |
| `VAN` | `Van`, net cargo/passenger van | `Minivan`, `MPV`, `Campervan` yalnız proje van örneğine yakınsa | `Bus`, büyük minibüs, ağır ticari araç |
| `PICK_UP` | `Pickup`, `Pick-Up`, `Pickup Truck`, `Ute` | generic `Truck` yalnız açık kasa + binek kabini varsa | `Big Truck`, `Dump Truck`, `Box Truck`, çekici, kamyon |
| `MICRO` | Net micro model veya micro/kei/quadricycle label | `City Car` yalnız Smart ForTwo, Toyota iQ, Renault Twizy, kei car vb. netse | Normal hatchback, compact car, supermini belirsizleri |
| `OPEN_WHEEL` | `Formula 1`, `F1`, `Formula`, `Single-seater`, `Open-wheel` | `Racecar` yalnız açık tekerlekliyse | GT, NASCAR, rally, Le Mans, kapalı tekerlekli yarış arabası |

### 13.3 İndirme önceliği: önce bunları indir/kontrol et

Aşağıdaki sıra, hem label güvenliği hem 30–40k ham hedefe ulaşma açısından önerilen sıradır. Önce `kaggle datasets files ...` ile boyut ve dosya yapısına bak; sonra indirmeye karar ver.

#### P0 — Kesin indirilecek / en güvenli çekirdek

| Öncelik | Kaynak | Komut | Alınacak label'lar | Dışarıda bırakılacaklar | Neden |
|---:|---|---|---|---|---|
| 1 | Cars Body Type Cropped | `kaggle datasets download -d ademboukhris/cars-body-type-cropped` | `Hatchback`, `Pick-Up`, `SUV`, `Sedan`, `VAN` | `Coupe`, `Convertible` | Ana ödev dokümanında referans; yaklaşık 7k, klasör label hazır, CC0. |
| 2 | Stanford Car Body Type Data | `kaggle datasets download -d mayurmahurkar/stanford-car-body-type-data` | İndirince klasör/CSV'de hedef 8'e uyanlar | Belirsiz/target dışı body type | Ana ödev dokümanında referans; önce dosya yapısı doğrulanacak. |
| 3 | VehicleType-10 | `kaggle competitions download -c vehicle-type-10-a-dataset-for-vehicle-body-type-classification` | `SUV`, `Van`, `Sedan`, `Hatchback`, `Pickup`, `Station Wagon` | Hedef dışı sınıflar | Küçük ama station wagon için temiz katkı. |
| 4 | F1 Image Classification Updated | `kaggle datasets download -d loveymishra/f1-image-classification-updated` | Tüm gerçek F1/formula görselleri → `OPEN_WHEEL` | F1 dışı/kapalı tekerlekli varsa at | Open-wheel sınıfı için ana kaynak. |
| 5 | Formula One Cars | `kaggle datasets download -d vesuvius13/formula-one-cars` | Gerçek F1/formula görselleri → `OPEN_WHEEL` | Takım/constructor label'ları hedef label değildir, hepsi `OPEN_WHEEL` altında birleşir | Open-wheel tamamlayıcı. |

#### P1 — 32k hedef için güçlü tamamlayıcılar

| Öncelik | Kaynak | Komut | Alınacak label'lar | Dışarıda bırakılacaklar | Neden |
|---:|---|---|---|---|---|
| 6 | Car Model Variants and Images / Car Models 3778 | `kaggle datasets download -d eimadevyni/car-model-variants-and-images-dataset` | CSV `body type` kolonundan güvenli eşleşenler: sedan, hatchback, SUV, wagon/estate, pickup, van; model adı net micro olanlar | Belirsiz body type, coupe, convertible, sports | 193k görsel + metadata; 30–40k hedefte en güçlü kota tamamlama kaynağı. |
| 7 | Vehicle Classification Dataset by iHasib | `kaggle datasets download -d ilhamhasib/vehicle-classification-dataset` | `Hatchback`, `Sedan`, `SUV`, `Pickup`; `MPV` sadece audit ile `VAN` olabilir | `Truck`, `Bus`, `Bike`, `CNG`, `Easy-Bike`; audit'siz `MPV` | 6.480 görsel, klasör/CSV label var; pickup label'ı generic truck'tan ayrı. |
| 8 | Stanford Car Dataset by classes folder | `kaggle datasets download -d jutrera/stanford-car-dataset-by-classes-folder` | Make-model-year lookup ile net sedan/SUV/van/hatchback/wagon/pickup | `coupe`, sports, belirsiz modeller | 16.185 görsel; body type yok ama class name üzerinden lookup yapılabilir. |
| 9 | Vehicle Images Dataset | `kaggle datasets download -d lyensoetanto/vehicle-images-dataset` | `Sport Utility Vehicle` → `SUV`, `Van` → `VAN`; `Sedan` yalnız coupe filtrelenirse; `City Car` yalnız micro/hatchback audit ile | `Truck`, `Big Truck`, audit'siz `City Car`, audit'siz `Sedan` içindeki coupe | 15.645 görsel ama label'lar karışık; seçici kullanılmalı. |

#### P2 — Hugging Face / domain çeşitliliği için küçük veya özel kaynaklar

| Öncelik | Kaynak | Komut | Alınacak label'lar | Dışarıda bırakılacaklar | Neden |
|---:|---|---|---|---|---|
| 10 | HF `kitrofimov/cbsc` | `huggingface-cli download kitrofimov/cbsc --repo-type dataset --local-dir data/raw/hf/kitrofimov_cbsc` | `hatchback`, `van`, `pickup`, `sedan`, `SUV` | `limousine`, `bus`, `sports`, `cabriolet`, generic `truck` | Küçük ama body-style label'lı; MIT. |
| 11 | HF `DrBimmer/vehicle-classification` | `huggingface-cli download DrBimmer/vehicle-classification --repo-type dataset --local-dir data/raw/hf/drbimmer_vehicle_classification` | Label yapısı incelenip sedan/SUV/truck vb. güvenli olanlar | Belirsiz/generic truck ve hedef dışı label'lar | 1k–10k aralığında imagefolder; lisans CC-BY-NC-4.0. |
| 12 | HF `iisc-aim/UVH-26` | `huggingface-cli download iisc-aim/UVH-26 --repo-type dataset --local-dir data/raw/hf/uvh_26` | Bbox crop ile net sedan/hatchback/SUV/van benzeri class'lar | Full frame, hedef dışı araçlar | Trafik/CCTV domain'i sağlar; object detection olduğu için crop pipeline gerekir. |

### 13.4 İlk turda indirmemeyi önerdiklerim

| Kaynak tipi | Neden ilk turda riskli? |
|---|---|
| Sadece `Car/Truck/Bus/Motorcycle` veren datasetler | Gövde tipi değil, üst seviye araç tipi. `Car` çok geniş, `Truck` pickup değil. |
| Genel `Vehicles Image Dataset` gibi 20 sınıflı web scrape setleri | `Car`, `Truck`, `Van` dışında body type ayrımı zayıf; label noise yüksek. |
| Generic `Racecar` datasetleri | Open-wheel ile GT/NASCAR/rally karışır. F1 kaynakları yetmezse audit ile kullanılır. |
| `Cars Collection Dataset` gibi çok büyük ama interior/metadata gürültülü setler | 60k+ görüntü var ama interior ve filename parse gürültüsü yüksek; P0/P1 yetmezse düşünülür. |

### 13.5 Tek tek bakmak için önce boyut/dosya listeleme komutları

İndirmeden önce şunları çalıştır:

```bash
kaggle datasets files -d ademboukhris/cars-body-type-cropped
kaggle datasets files -d mayurmahurkar/stanford-car-body-type-data
kaggle datasets files -d eimadevyni/car-model-variants-and-images-dataset
kaggle datasets files -d ilhamhasib/vehicle-classification-dataset
kaggle datasets files -d lyensoetanto/vehicle-images-dataset
kaggle datasets files -d jutrera/stanford-car-dataset-by-classes-folder
kaggle datasets files -d loveymishra/f1-image-classification-updated
kaggle datasets files -d vesuvius13/formula-one-cars
```

Competition için:

```bash
kaggle competitions files -c vehicle-type-10-a-dataset-for-vehicle-body-type-classification
```

Hugging Face için hızlı metadata kontrolü:

```bash
huggingface-cli repo info kitrofimov/cbsc --repo-type dataset
huggingface-cli repo info DrBimmer/vehicle-classification --repo-type dataset
huggingface-cli repo info iisc-aim/UVH-26 --repo-type dataset
```

### 13.6 Pratik ilk indirme seti

Eğer tek tek indirip boyutlara bakacaksan önerilen sıra:

1. `ademboukhris/cars-body-type-cropped`
2. `mayurmahurkar/stanford-car-body-type-data`
3. `vehicle-type-10-a-dataset-for-vehicle-body-type-classification`
4. `loveymishra/f1-image-classification-updated`
5. `vesuvius13/formula-one-cars`
6. `ilhamhasib/vehicle-classification-dataset`
7. `eimadevyni/car-model-variants-and-images-dataset`
8. `jutrera/stanford-car-dataset-by-classes-folder`
9. `lyensoetanto/vehicle-images-dataset`
10. `kitrofimov/cbsc`
11. `DrBimmer/vehicle-classification`
12. `iisc-aim/UVH-26`

İlk 6 kaynak indirildikten sonra sınıf sayıları kontrol edilmeli. Eğer toplam temiz sayı 20k civarında kalırsa 7–9. kaynaklar açılır. Eğer `OPEN_WHEEL` 3.5k altındaysa F1 kaynaklarına ek Roboflow/open-wheel kaynakları aranır. Eğer `MICRO` 3.2k altındaysa Car Model Variants + Stanford/VMMR lookup üzerinden net micro modeller hedeflenir.
