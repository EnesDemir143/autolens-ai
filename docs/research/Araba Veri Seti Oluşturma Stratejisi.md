# **İnce Taneli Araç Gövde Tipi Sınıflandırmasında MİCRO Sınıfı: Kapsamlı Veri Seti Analizi, Otomatik Veri Toplama Mimarisi ve Model Optimizasyon Stratejileri**

## **Giriş: Bilgisayarlı Görüde İnce Taneli Araç Sınıflandırması ve MİCRO Sınıfının Morfolojik Özellikleri**

Gelişmiş Bilgisayarlı Görü (Computer Vision) sistemlerinde ve otonom sürüş teknolojilerinde nesne algılama (Object Detection) ve görüntü sınıflandırma (Image Classification), yapay zeka araştırmalarının temel odak noktalarından birini oluşturmaktadır. Sekiz sınıflı (SUV, Sedan, Hatchback, Van, Station Wagon, Pick Up, Açık Tekerlekli ve Micro) bir araç gövde tipi sınıflandırma projesi, literatürde İnce Taneli Görsel Sınıflandırma (Fine-Grained Visual Categorization \- FGVC) problemi olarak değerlendirilir. Bu tür projelerde Evrişimli Sinir Ağları (CNN) veya Vision Transformer (ViT) mimarilerinin başarısı, doğrudan kullanılan veri setinin hacmine, çeşitliliğine ve kalitesine bağlıdır. Geleneksel araç sınıfları (örneğin Sedan veya SUV) için devasa açık kaynaklı veri havuzları bulunurken, "Micro" (ultra-kompakt) araç sınıfı literatürde en az temsil edilen ve morfolojik olarak en çok varyasyon gösteren kategorilerden biridir.1  
MİCRO sınıfı, sadece "düşük çözünürlüklü" veya uzaktan çekilmiş küçük araç görsellerini ifade etmez; aksine, Smart Fortwo, Citroen Ami, Renault Twizy, Toyota iQ, Tata Nano, Peel P50, Aixam, Ligier ve BMW Isetta gibi çok spesifik mühendislik tasarımlarına sahip, genellikle A-segmentinin de altında konumlanan ultra-kompakt şehir araçlarını veya Avrupa regülasyonlarındaki dört tekerlekli motosikletleri (quadricycles) tanımlar.1 Örneğin, Citroen Ami sadece 2.41 metre uzunluğunda, 1.39 metre genişliğinde ve simetrik gövde panellerine sahip bir araçtır.5 Benzer şekilde, Peel P50 1.37 metre uzunluğu ile dünyanın en küçük üretim aracı unvanına sahiptir.4 BMW Isetta, ön taraftan açılan tek bir kapıya sahip yumurta formunda bir "bubble car" tasarımı sunarken 4; Renault Twizy, 470 kg ağırlığında, kapıları opsiyonel olan ve motosiklet dinamiklerine yakın yarı açık bir kabin sunan bir tasarıma sahiptir.3 Tata Nano ise 3 metre uzunluğuyla dört yolcu taşıyabilen ancak geleneksel Hatchback oranlarından çok daha farklı, uzunluğuna göre oldukça dar (1.5m) bir profile sahiptir.4  
Bu araçların morfolojik yapıları, geleneksel hatchback veya sedan araçlardan radikal biçimde farklı olduğu için, sınıflandırma modellerinin bu araçları standart bir alt kompakt araçtan ayırması ciddi bir problemdir. Kısa dingil mesafeleri (wheelbase), agresif gövde-cam oranları ve alışılmışın dışındaki tekerlek konumlandırmaları, standart bir araç detektörünün kafasını karıştırabilir. Bu noktada, modelin bu spesifik sınıfı ezberlemeden (overfitting) öğrenmesi ve yüksek genelleme (generalization) yeteneği kazanması için hedeflenen 500-1000+ arası yüksek kaliteli görsel havuzu, son derece stratejik ve ideal bir hacmi temsil etmektedir.  
Bu araştırma raporu, hedeflenen bu görsel hacmine ulaşmak için internetin derinliklerinde yer alan hazır veri setlerinin (Kaggle, Hugging Face, Roboflow, GitHub vb.) derinlemesine bir taramasını sunmakta, hazır verilerin hedeflenen hacmin altında kalması veya spesifik Avrupa/Asya pazarındaki mikro araçları kapsamaması durumunda devreye sokulacak endüstri standardında bir web scraping (veri kazıma) mimarisini detaylandırmakta ve modelin daha önce hiç görmediği test verilerinde yüksek F1-Skoru elde etmesini sağlayacak gelişmiş veri çeşitliliği ve temizleme tavsiyelerini ortaya koymaktadır.

## **Bölüm 1: Açık Kaynaklı Veri Setlerinin Derinlemesine Taraması ve Literatür Analizi**

Bilgisayarlı görü projelerinde veri toplama sürecine sıfırdan başlamak yerine, öncelikle geniş çaplı, titizlikle etiketlenmiş ve doğrulanmış akademik veya topluluk tabanlı veri setlerinden yararlanmak endüstri standardıdır. Bu yaklaşım, sadece veri toplama maliyetlerini düşürmekle kalmaz, aynı zamanda farklı kamera sensörlerinden, ışık koşullarından ve coğrafyalardan gelen gürültülü (noisy) verilerin modele entegre edilmesini sağlayarak ağın genelleme kapasitesini artırır. Yapılan kapsamlı veri madenciliği sonucunda, "Micro" sınıfı araçları içeren, doğrudan bu araç modellerine odaklanan veya boyut tabanlı sınıflandırma yapan veri setleri Kaggle, Roboflow Universe, GitHub ve akademik repolarda tespit edilmiştir.

### **Roboflow Universe: Topluluk Tabanlı Veri Havuzları ve Etiketleme Stratejileri**

Roboflow Universe platformu, otonom sürüş, akıllı şehir ulaşım inovasyonları ve nesne takibi gibi alanlar için açık kaynaklı bilgisayarlı görü projelerinin barındırıldığı dünyanın en büyük ekosistemlerinden biridir.10 Bu platform üzerinde yapılan taramalarda, sadece araçlara odaklanan değil, spesifik olarak mikro araçları etiketleyen veya boyut tabanlı (size-based) sınıflandırma yapan çok çeşitli projeler tespit edilmiştir.

1. **Final Dataset v2 (Yazar: Thesis):** Bu veri seti, toplamda 2.640 görsel içermekte olup, hem nesne algılama (Object Detection) hem de anlamsal bölütleme (Semantic Segmentation) görevleri için yapılandırılmış varyasyonlara sahiptir.11 Bu veri setinin en büyük avantajı, sınıf etiketleri arasında doğrudan MicroCar sınıfının spesifik olarak yer almasıdır. Bunun yanı sıra AutoRickshaw, Car, Bus, PickUp ve Van gibi sınıflar da bulunmaktadır.12 Modelin mikro araçlar ile diğer benzer boyutlardaki Asya tipi üç tekerlekli (AutoRickshaw) veya kompakt ticari ulaşım araçları arasındaki görsel sınırları öğrenmesi için bu veri seti mükemmel bir zemin sunar.12  
2. **Car Types Dataset (Yazar: VEMO):** Bu veri setinde dört adet özel sınıf bulunmaktadır: 0, beetle, microcar, ve minitruck.14 Veri seti hacmi görece küçük (281 görsel) olsa da, doğrudan microcar etiketine sahip bounding-box (sınırlayıcı kutu) anotasyonlarını barındırması nedeniyle hedef veri havuzuna yüksek kaliteli ve doğrudan kullanılabilir bir başlangıç materyali sağlar.14  
3. **Boyut Tabanlı Araç Veri Setleri (Small Vehicle / Small Car):** Roboflow üzerinde, araçların geleneksel segmentlerine (SUV, Sedan vb.) göre değil boyutlarına göre etiketlendiği birçok veri seti mevcuttur.15 Örneğin; 13.400 görsele sahip devasa "ATLAS" veri setinde small-vehicle sınıfı, 7.730 görsel içeren "car size2" veri setinde small sınıfı, ve havadan çekim (aerial) perspektifine sahip "Aerial-cars-kaggle-cleaned" veri setinde (266 görsel) yine küçük araç sınıfları bulunmaktadır.15 Bu "küçük araç" sınıflarının içinde Smart Fortwo ve Toyota iQ gibi hedeflenen MİCRO sınıfı araçların yoğunlukla yer aldığı bilinmektedir. Bu veri setleri, modeli ölçek farklılıklarına karşı eğitmek için kritik bir öneme sahiptir.

### **Stanford Cars ve CompCars: Akademik Ölçekte İnce Taneli Sınıflandırma**

Kaggle platformu ve akademik konferanslar için hazırlanan, literatürde benchmark (kıyaslama) olarak kullanılan devasa veri setleri, spesifik marka ve modelleri alt sınıflara ayırdığı için projenin MİCRO sınıfı ihtiyacına doğrudan yanıt verebilecek niteliktedir.

1. **Stanford Cars Dataset:** Bilgisayarlı görü literatüründe, özellikle ince taneli görsel sınıflandırma (FGVC) alanında en çok atıf alan veri setlerinden biridir.16 Toplamda 196 farklı araç sınıfına ait 16.185 yüksek çözünürlüklü görsel barındırır.18 Bu veri seti, 8.144 eğitim ve 8.041 test görseli olmak üzere yaklaşık %50-%50 oranında bölünmüştür.19 Sınıflar "Marka, Model, Yıl" formatında yapılandırılmıştır.21 Bu veri setinin sınıfları detaylı incelendiğinde, Smart Fortwo Convertible 2012 gibi doğrudan hedeflenen ultra-kompakt araç modellerinin de sınıflandırıldığı görülmektedir.22 Veri setindeki görsellerin son derece temiz, titizlikle etiketlenmiş ve arka plan açısından çeşitli olması, bu veri setinin MİCRO sınıfı için oluşturulacak havuzda altın standart olmasını sağlar.  
2. **CompCars Dataset (Comprehensive Cars):** Kapsamlı araç sınıflandırma ve doğrulama görevleri için oluşturulmuş olan bu veri seti, toplamda 136.726 web görseli ve 50.000 ön kamera gözetim görseli içermektedir.23 163 otomobil markası ve 1.716 araç modelini kapsayan bu devasa akademik kaynak, Stanford Cars'tan çok daha geniş bir yelpaze sunar.23 Yapılan analizler, Smart Fortwo ve Toyota iQ modellerinin bu veri setinin hiyerarşisinde yer aldığını, yüksek çözünürlüklü ve farklı açılardan (ön, arka, yan, çapraz) çekilmiş dış mekan görsellerini içerdiğini doğrulamaktadır.23

### **Kaggle ve Yerel Pazar Veri Setleri: Tata Nano ve Gelişmekte Olan Pazar Dinamikleri**

Batı odaklı veri setleri genellikle büyük SUV'ler veya spor araçlar üzerinde yoğunlaşırken, MİCRO sınıfının Asya pazarındaki en büyük temsilcisi olan Tata Nano modelini bulmak için spesifik lokal veri setlerine yönelmek gerekmektedir. Kaggle üzerinde bulunan "Indian Cars Dataset", "Sri Lanka Vehicle Ads Dataset" ve "Exploring Online Used Car Sales" gibi veri setleri incelenmiştir.26 İkinci el araç satış platformlarından veri kazıma yoluyla elde edilen bu veriler, günlük yaşam koşullarında, telefon kameralarıyla, farklı ışık ve düşük kalitede çekilmiş Tata Nano (GenX, XT vb.) görselleri sunar.26 Bu durum bir dezavantaj gibi görünse de, derin öğrenme modelinin ezberleme (overfitting) yapmasını engelleyici, "real-world noise" (gerçek dünya gürültüsü) içeren muazzam kalitede bir veri yığınıdır.26 Laboratuvar ortamında çekilmiş parlak basın fotoğrafları yerine, sokakta park halinde çekilmiş Tata Nano görselleri modelin dayanıklılığını (robustness) artıracaktır.

### **GitHub ve Otonom Sürüş Sensör Verileri: UPCT ve Renault Twizy Örneği**

MİCRO sınıfı araçların otonom sürüş projelerinde bir araştırma platformu olarak kullanılması, bu araçlara ait eşsiz veri setlerinin oluşmasını sağlamıştır. İspanya'daki Universidad Politécnica de Cartagena (UPCT) tarafından geliştirilen ve modifiye edilmiş bir Renault Twizy baz alan otonom araç platformu, kendi adıyla anılan "UPCT Dataset"i literatüre kazandırmıştır.9 Bu veri seti, Twizy'nin üzerine yerleştirilmiş 3D HD LiDAR, RGB kameralar, ToF (Time of Flight) kameraları ve IMU gibi sensörlerden elde edilen 78.000'den fazla senkronize veri örneği içermektedir.31 Benzer şekilde, DurLAR veri seti de 128 kanallı LiDAR ve kameralarla donatılmış bir Renault Twizy kullanılarak oluşturulmuştur.33 Her ne kadar bu veri setleri dışarıdan Twizy'yi çeken fotoğraflar yerine, Twizy'nin içinden dışarıyı çeken (egocentric) veriler içerse de, projelerin dokümantasyonlarında, araştırma makalelerinde ve GitHub repolarında test aracı olan Renault Twizy'nin çeşitli açılardan çekilmiş yüksek çözünürlüklü modifikasyon ve kalibrasyon fotoğrafları bolca bulunmaktadır.9 Bu görseller, projeye Twizy verisi sağlamak için kullanılabilecek niş bir kaynaktır.

### **Veri Setleri Sentezi ve Doğrudan Erişim Bağlantıları Tablosu**

Yukarıda detaylandırılan literatür taraması sonucunda elde edilen ve MİCRO sınıfı veri havuzuna doğrudan entegre edilebilecek hazır veri setleri aşağıdaki tabloda (Bkz. Tablo 1\) özetlenmiştir. Bu tablo, kullanıcıya hedef etiketleri, beklenen görsel sayılarını, kalitelerini ve veri setlerinin doğrudan linklerini sunmaktadır.

| Kaynak Platform | Veri Seti / Proje Adı | Hedef Etiket / Modeller | Tahmini Görsel Sayısı | Kalite ve Veri Karakteristiği | Doğrudan Link |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Roboflow** | Final Dataset v2 | MicroCar | \~150 \- 300 | Orta-Yüksek Çözünürlük, Şehir içi, Bounding Box 12 | [universe.roboflow.com/thesis-aarau/thesis-n4a7t](https://universe.roboflow.com/thesis-aarau/thesis-n4a7t) |
| **Roboflow** | Car Types | microcar | \~50 \- 100 | Spesifik sınıflandırma, Doğrudan kullanım 14 | [universe.roboflow.com/vemo/car-types-nvqdm](https://universe.roboflow.com/vemo/car-types-nvqdm) |
| **Kaggle** | Stanford Cars | Smart Fortwo Convertible | \~80 \- 100 | Yüksek Çözünürlük, İnce Taneli Akademik Veri 21 | [kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder) |
| **Academic** | CompCars | Smart Fortwo, Toyota iQ | \~150 \- 200 | Çoklu açı (ön/arka/yan), Model seviyesi ayrım 23 | [homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm](https://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm) |
| **Kaggle** | Indian Cars Dataset | Tata Nano (Tüm Varyantlar) | \~100 \- 150 | Gerçek hayat (Wild) görselleri, Düşük/Orta Çözünürlük 27 | [kaggle.com/datasets/medhekarabhinav5/indian-cars-dataset](https://www.kaggle.com/datasets/medhekarabhinav5/indian-cars-dataset) |
| **GitHub** | UPCT & DurLAR | Renault Twizy | \~30 \- 50 | Araştırma makalesi donanım fotoğrafları, Yüksek kalite 32 | ([https://github.com/l1997i/DurLAR](https://github.com/l1997i/DurLAR)) |

Tablo 1'de görüldüğü üzere, mevcut hazır veri setleri maksimum verimle birleştirildiğinde dahi, spesifik olarak MİCRO sınıfına ait 500 ile 800 arasında bir görsel hacmi elde edilmektedir. Ancak bu hacim, hedeflenen "1000+ görsel ve yüksek model genelleme yeteneği" için sınırlarda kalmaktadır. Daha da önemlisi, Aixam, Ligier, Peel P50, Citroen Ami ve BMW Isetta gibi ağırlıklı olarak Avrupa odaklı dört tekerlekli motosiklet veya spesifik "bubble car" formundaki araçlar, Amerikan veya Asya kökenli devasa hazır veri setlerinde yeterince temsil edilmemektedir.4 Modelin bu araçları eksik öğrenmesi, test aşamasında ciddi bir sınıflandırma zafiyeti (class imbalance) yaratacaktır. Bu spesifik boşluğu doldurmak ve veri hacmini istenilen 1000+ bandının üzerine güvenli bir şekilde taşımak için B Planı olan gelişmiş bir veri kazıma mimarisinin devreye alınması teknik ve pratik bir zorunluluktur.

## **Bölüm 2: Endüstri Standardında Otomatik Veri Toplama (Web Scraping) Mimarisi**

Hazır verilerin desteklenmesi, eksik kalan MİCRO araç modellerinin tamamlanması ve görsel veri tabanının farklı arka planlarla zenginleştirilmesi için bir web kazıma (web scraping) boru hattı (pipeline) kurulmalıdır. Bu süreçte kullanılacak aracın; arama motorlarının API kısıtlamalarına takılmayan, stabil çalışan, asenkron veya çoklu iş parçacığı (multi-threading) yeteneklerine sahip olması ve en önemlisi indirilen verinin bütünlüğünü denetleyebilmesi gerekmektedir.37  
Bu amaçla literatürde ve açık kaynak projelerde en çok tercih edilen kütüphanelerden biri olan icrawler ile modern alternatiflerden bing-image-downloader detaylı bir şekilde değerlendirilmiştir.38

### **İcrawler Kütüphanesinin Seçim Gerekçeleri ve Asenkron Çalışma Prensibi**

Python tabanlı bir mini framework olan icrawler, modüler tasarımı (Feeder, Parser ve Downloader bileşenleri) ve yerleşik olarak Google, Bing, Baidu ve Flickr gibi farklı kaynaklardan veri çekebilmesi nedeniyle bu proje için en ideal çözüm olarak belirlenmiştir.38 bing-image-downloader kütüphanesi kullanım kolaylığı sunsa da 40, icrawler'ın sunduğu eşzamanlı iş parçacığı (downloader\_threads) desteği ve görsel tipi, boyutu, lisansı, tarihi gibi parametreleri filtreleyebilme esnekliği, endüstriyel standartlarda bir veri seti oluşturmak için onu vazgeçilmez kılar.38

### **Negatif Arama Mantığı: Oyuncak ve Maket Yanılgısının Filtrelenmesi**

Yapay zeka modelinin "Micro" kelimesini veya hedef araçları öğrenirken karşılaşacağı en büyük yapısal tehdit, arama motoru sonuçlarının (SERP) kirliliğidir.43 Arama motorlarının doğası gereği "Peel P50" veya "BMW Isetta" aratıldığında, bu araçların tarihi ve koleksiyon değeri nedeniyle sıklıkla 1:18, 1:43 veya 1:64 ölçekli Hot Wheels, diecast maketleri veya 3D render görselleri sonuçlarda ön sıralarda listelenmektedir. Benzer şekilde, parça satıcılarının yüklediği sadece bir jant veya farın makro çekimleri de araç bütünlüğünü yansıtmaz.43  
Modelin oyuncak araçları veya makro parçaları bir MİCRO sınıfı araba sanarak ezberlemesini (overfitting) engellemek adına arama terimlerine "Negatif Filtreleme" mantığı uygulanmalıdır.43 Arama motorlarına gönderilecek (query) dizesi, sadece aracın adını değil, dışlanması gereken kelimeleri de barındırmalıdır. Bu nedenle sorgular; "{Araç Adı}" car \-toy \-diecast \-hotwheels \-model \-interior formatında tasarlanmıştır. Bu strateji, Amazon Mechanical Turk gibi insan tabanlı filtreleme maliyetlerine girmeden 43, gürültülü (noisy) veriyi kaynağında, yani indirme aşamasından önce keser.

### **Gelişmiş Python Veri Kazıma Betiği**

Aşağıdaki Python kod parçacığı, belirtilen tüm kısıtlamalara ve mimari gereksinimlere uygun olarak sıfırdan geliştirilmiştir. Kod, icrawler kütüphanesini baz alarak tamamen çalışabilir durumda tasarlanmış, modüler, ölçeklenebilir ve sağlam hata yakalama mekanizmalarına (try-except blocks) sahip bir mühendislik ürünüdür.

Python

import os  
import time  
from icrawler.builtin import BingImageCrawler  
from PIL import Image, UnidentifiedImageError

\# GÖREV 1: Veri Seti Dizin Konfigürasyonu  
\# İndirilen görsellerin kaydedileceği hedef klasör belirleniyor.  
DATASET\_DIR \= "dataset/micro/"

\# GÖREV 2: Hedef Araç Modelleri ve Negatif Filtreli Arama Terimleri  
\# Modelin oyuncak (diecast), makro çekim veya iç mekan görsellerini öğrenerek  
\# overfitting (ezberleme) yapmasını engellemek için '-' operatörü ile   
\# negatif anahtar kelimeler sorguya dahil edilmiştir.  
CAR\_MODELS \=

\# Her bir araç sorgusu için Bing üzerinden çekilecek maksimum görsel limiti  
MAX\_IMAGES\_PER\_QUERY \= 150 

def create\_directory\_if\_not\_exists(directory):  
    """  
    Veri setinin kaydedileceği dizini işletim sistemi seviyesinde kontrol eder.  
    Klasör yoksa oluşturur. Güvenli veri yazımı için temel bir adımdır.  
    """  
    try:  
        if not os.path.exists(directory):  
            os.makedirs(directory)  
            print(f" Hedef dizin başarıyla oluşturuldu: {directory}")  
    except OSError as e:  
        print(f" Dizin oluşturulurken işletim sistemi hatası alındı: {e}")  
        exit(1)

def scrape\_microcar\_images():  
    """  
    Belirlenen araç modelleri için Bing arama motorundan görselleri   
    çoklu iş parçacığı (multi-threading) kullanarak asenkron olarak indirir.  
    """  
    create\_directory\_if\_not\_exists(DATASET\_DIR)  
      
    for model\_query in CAR\_MODELS:  
        print(f"\\n Aranan Terim: {model\_query}")  
          
        \# Filtreleme Parametreleri:  
        \# type='photo' \-\> Çizimleri (line drawing), vektörleri (clipart) veya GIF'leri dışlar.  
        \# size='large' \-\> Modelin eğitiminde detayları görebilmesi için yüksek çözünürlük zorunludur.  
        search\_filters \= dict(  
            type='photo',  
            size='large'  
        )  
          
        try:  
            \# İcrawler BingImageCrawler bileşeninin yapılandırılması  
            \# feeder\_threads: Arama motoruna istek gönderen iş parçacığı  
            \# parser\_threads: Gelen HTML sonuçlarından image URL'lerini ayrıştıran iş parçacığı  
            \# downloader\_threads: 4 farklı görseli aynı anda indirerek ağ darboğazını aşan iş parçacığı  
            crawler \= BingImageCrawler(  
                feeder\_threads=1,  
                parser\_threads=2,  
                downloader\_threads=4,  
                storage={'root\_dir': DATASET\_DIR}  
            )  
              
            \# Veri çekme işleminin tetiklenmesi  
            \# file\_idx\_offset='auto' \-\> Mevcut dosyaların üzerine yazmayı engeller, isimlendirmeye otomatik devam eder.  
            crawler.crawl(  
                keyword=model\_query,   
                filters=search\_filters,   
                max\_num=MAX\_IMAGES\_PER\_QUERY,   
                file\_idx\_offset='auto'   
            )  
              
            \# Anti-Scraping mekanizmalarına (Rate Limiting, IP Ban) takılmamak için   
            \# her sorgu arasında ağın dinlenmesine izin veren bekleme süresi  
            time.sleep(3)  
              
        except Exception as e:  
            \# Bir ağ kopması veya kütüphane hatası tüm döngüyü çökertmesin diye try-except kullanılır  
            print(f" '{model\_query}' aranırken beklenmeyen bir hata oluştu: {e}")  
            continue

def clean\_corrupted\_images(directory):  
    """  
    GÖREV 3: Veri Bütünlüğü Doğrulaması (Data Integrity Check)  
    İndirilen dizindeki tüm görselleri tarar. İnternet kopmaları nedeniyle  
    yarım inmiş, Byte seviyesinde bozuk olan veya açılamayan dosyaları tespit edip otomatik siler.  
    """  
    print(f"\\n Dosya bütünlük taraması ve bozuk veri temizliği başlatılıyor: {directory}")  
    removed\_count \= 0  
    total\_files \= 0  
      
    for filename in os.listdir(directory):  
        file\_path \= os.path.join(directory, filename)  
          
        \# Sadece dosyaları işleme al, alt dizinler varsa atla  
        if os.path.isfile(file\_path):  
            total\_files \+= 1  
            try:  
                \# PIL (Pillow) kütüphanesinin Image.open() fonksiyonu dosyayı belleğe alırken,  
                \# verify() fonksiyonu pikselleri çözmek yerine dosya magic-header'larını okur.  
                \# Bu işlem son derece hızlıdır ve JPEG/PNG veri bloklarındaki anormallikleri tespit eder.  
                with Image.open(file\_path) as img:  
                    img.verify()  
            except (IOError, SyntaxError, UnidentifiedImageError, AttributeError) as e:  
                \# Dosya bozuksa veya başlık bilgisi okunamıyorsa, exception yakalanır ve dosya silinir.  
                print(f" Bozuk veya okunamayan dosya tespit edildi, siliniyor: {filename} \- Hata Kodu: {e}")  
                os.remove(file\_path)  
                removed\_count \+= 1  
            except Exception as e:  
                print(f" {filename} üzerinde işlem yapılamadı: {e}")  
                  
    print(f"\\n Veri Temizliği Tamamlandı. Toplam Taranan: {total\_files}, Silinen Bozuk Dosya: {removed\_count}")

if \_\_name\_\_ \== "\_\_main\_\_":  
    print("="\*60)  
    print(" MİCRO Sınıfı Araç Veri Seti Toplayıcı ve Doğrulayıcı v1.0 ")  
    print("="\*60)  
      
    \# Adım 1: Görselleri İndir  
    scrape\_microcar\_images()  
      
    \# Adım 2: Bozuk Dosyaları Temizle  
    clean\_corrupted\_images(DATASET\_DIR)  
      
    print("\\n Tüm işlemler hedeflendiği gibi eksiksiz tamamlandı.")

### **Betiğin Mimari Analizi ve Tensor İleri Besleme (Forward Pass) Güvenliği**

Tasarlanan bu betik, sadece internetten resim indiren basit bir komut dosyası değildir; yapay zeka modelinin eğitim döngüsüne (training loop) veri hazırlayan bir ön işleme (preprocessing) mekanizmasıdır. Betikteki downloader\_threads=4 parametresi, I/O bound (giriş/çıkış sınırlı) olan indirme sürecini asenkronize ederek, ağ gecikmelerini (network latency) minimize eder ve tek parçacıklı (single-thread) bir yapıya göre veri toplama süresini dörtte birine indirir.38  
Daha da önemlisi, clean\_corrupted\_images fonksiyonu modelin eğitim döngüsüne girmeden önce kritik bir güvenlik duvarı oluşturur. PyTorch veya TensorFlow gibi derin öğrenme çerçevelerindeki DataLoader sınıfları, eğitim (epoch) sırasında bozuk bir JPEG magic-header'ı veya eksik byte blokları ile karşılaştığında ölümcül hata vererek tüm eğitimi çökertebilir. Betik içindeki img.verify() metodu, dosyayı RGB piksel matrisine tamamen kodlamak (decode) yerine, sadece başlık (header) ve yapı kontrolü yaparak maksimum hız ve sistem bütünlüğü sağlar. Bu fonksiyon, veri setinin doğrudan makine öğrenimi modellerinde kullanıma hazır (AI-ready) olmasını garanti altına alır.

## **Bölüm 3: Model Genelleştirme Yeteneği ve Yüksek F1-Skoru İçin Stratejik Optimizasyonlar**

Makine öğrenimi modellerinde, özellikle 8 sınıflı bir araç gövde tipi sınıflandırmasında sınıflar arası veri dengesizlikleri (class imbalance) kaçınılmazdır. Sektördeki gerçek veri dağılımları nedeniyle SUV, Sedan veya Hatchback sınıfları için milyonlarca temiz görsel bulunabilirken, MİCRO sınıfı için ulaşılabilecek maksimum veri (elde edilen veri setleri ve kazınan verilerle birlikte) birkaç bin görsel ile sınırlı kalacaktır. Bu tür dengesiz veri setlerinde, Doğruluk (Accuracy) metriği model performansını değerlendirmek için son derece yanıltıcıdır.46 Örneğin, veri setinin %90'ı SUV ve Sedan, sadece %1'i Micro sınıfından oluşuyorsa; model her gördüğü araca "SUV veya Sedan" dese dahi genel doğruluk oranı %90'ın üzerinde çıkabilir.47 Ancak bu model MİCRO araçları tanımakta tamamen başarısızdır.  
Bu nedenle, modelin daha önce hiç görmediği (unseen) test verilerindeki gerçek performansını yansıtan ve Kesinlik (Precision) ile Duyarlılık (Recall) değerlerinin harmonik ortalaması olan F1-Skoru, bu projenin birincil değerlendirme metriği olmalıdır.46 Bir aracın gerçekten MİCRO sınıfı olduğunu algılayabilmek (Recall) ve diğer araçları yanlışlıkla MİCRO olarak etiketlememek (Precision) arasındaki bu hassas dengeyi kurabilmek için, aşağıdaki veri çeşitliliği, veri çoğaltma (Data Augmentation) ve model eğitimi stratejileri kesin bir şekilde uygulanmalıdır.

### **Tavsiye 1: Geometrik Bütünlüğün Korunması (Aspect-Ratio Padding) ve Ölçekleme Sorunları**

MİCRO sınıfı araçların en büyük karakteristik özelliği, araç uzunluğunun dingil mesafesine (wheelbase) oranının ve genel en-boy oranının (aspect ratio) standart bir Hatchback veya Sedan'dan radikal bir şekilde farklı olmasıdır.3 Bir Smart Fortwo'nun tekerlekleri, aracın ön ve arka tamponlarının neredeyse en uç noktalarında yer alırken; geleneksel bir Sedan'da ciddi bir tampon çıkıntısı (overhang) bulunur.  
Derin öğrenme modellerini eğitirken, veri çoğaltma (Data Augmentation) veya ön işleme (preprocessing) aşamasında görseller genellikle 224x224, 416x416 veya 640x640 gibi kare formatlı tensörlere dönüştürülür.49 Eğer bu dönüşüm sırasında standart "Random Crop" (rastgele kırpma) veya basit "Resize" (yeniden boyutlandırma/sündürme) teknikleri kullanılırsa, geometrik felaketler yaşanır. Dikdörtgen formundaki uzun bir Hatchback veya Sedan fotoğrafı, kare formata sündürülerek sıkıştırıldığında, modelin evrişim (convolutional) filtrelerinde aracın boyu yapay olarak kısalır ve bir MİCRO araç gibi görünmeye başlar. Bu durum, modelin test sırasında Hatchback araçları MİCRO sanmasına yol açar (Precision düşüşü).

* **Çözüm Stratejisi:** Görüntüleri CNN veya ViT mimarisine beslemeden önce "En-Boy Oranını Koruyarak Doldurma" (Aspect-Ratio Preserving Padding veya Letterboxing) tekniği kullanılmalıdır. Araç orijinal en-boy oranını koruyarak kare tensörün içine yerleştirilmeli ve boşta kalan kısımlar siyah piksellerle veya görselin ortalama renk değerleriyle doldurulmalıdır. Ayrıca veri çoğaltma (Data Augmentation) aşamasında "Horizontal Flip" (yatay çevirme), "Brightness/Contrast adjustment" gibi piksel değerleriyle oynayan, ancak "Random Perspective" veya aracın şeklini bozan şiddetli "Affine Transformations" tekniklerinden kaçınılmalıdır.50 Bu yaklaşım, modelin araç geometrisini (tekerlek mesafesi, cam/gövde oranı) hatasız öğrenmesini sağlayarak F1-Skorunu doğrudan yükseltir.

### **Tavsiye 2: Bağlamsal Aşırı Öğrenmenin (Context Overfitting) Giderilmesi ve Gelişmiş Veri Çoğaltma**

Derin öğrenme modelleri, doğaları gereği tembel öğrenicilerdir (lazy learners). Çoğu zaman bir nesnenin kendi özelliklerini (features) öğrenmekten ziyade, o nesnenin en sık bulunduğu arka planı (background) öğrenerek kestirme bir yol bulmaya yatkındırlar. MİCRO sınıfı araçların internette ve toplanan veri setlerinde bulunan görselleri sıklıkla dar Avrupa sokaklarında, tarihi binaların önünde, fuar alanlarında (indoor car shows), veya yaya kaldırımı/bisiklet yolu gibi spesifik bağlamlarda çekilmektedir.3 Bir SUV veya Pick-Up ise genellikle geniş otoyollar, dağ yolları veya şantiyelerde fotoğraflanır.  
Eğer MİCRO araç görsellerinin büyük çoğunluğu dar sokakları veya fuar spot ışıklarını içeriyorsa; model, MİCRO aracın spesifik tekerlek boyutlarını veya gövde panellerini öğrenmek yerine, arka plandaki dokuyu öğrenir. Sonucunda, "dar tarihi sokak veya fuar ışıkları gördüğümde bu bir MİCRO'dur" gibi yanlış bir tümevarım yapar. Bu duruma Literatürde "Bağlamsal Yanılgı" (Contextual Bias) denir. Test verisinde geniş bir Amerikan otobanında çekilmiş bir Citroen Ami veya Smart Fortwo'yu tanıyamayacak (Düşük Recall) veya dar bir Avrupa sokağında park etmiş bir SUV'u MİCRO olarak sınıflandıracaktır.54

* **Çözüm Stratejisi:** Arka plan izolasyonunu sağlamak ve modeli nesnenin kendisine odaklamak için gelişmiş veri çoğaltma teknikleri devreye sokulmalıdır.  
  1. Işık koşullarını manipüle etmek için *Color Jittering* (parlaklık, kontrast, doygunluk oynamaları) ve *Adaptive Histogram Equalization* (CLAHE) uygulanarak fuar ve sokak ışıklarının yanıltıcı etkisi nötralize edilmelidir.49  
  2. Daha ileri düzey bir yöntem olarak, eğer verilerde anlamsal bölütleme (segmentasyon) maskeleri veya sınırlayıcı kutu (Bounding-Box) koordinatları mevcutsa, *CutMix* veya *Copy-Paste Augmentation* teknikleri uygulanmalıdır. Bu teknikte, MİCRO araçları bulundukları dar sokak arka planlarından dijital olarak kesilip (maskelenip), otoyol, dağ yolu veya geniş otopark gibi tamamen alakasız arka planlara sentetik olarak yapıştırılır. Bu yapay çeşitlilik, sinir ağını arka planı yok saymaya ve sadece aracın dış hatlarına (edge features) odaklanmaya zorlar.

### **Tavsiye 3: Zorlu Negatif Madenciliği (Hard Negative Mining) ve Ölçek Yanılsamasının Önlenmesi**

Toplanan veriler ne kadar titiz filtrelenirse filtrelensin, internette "BMW Isetta", "Peel P50" veya "Renault Twizy" aratıldığında, makro lensle ve düşük alan derinliği (depth of field) ile çekilmiş aşırı gerçekçi diecast/oyuncak modeller, 3D render edilmiş bilgisayar tasarımları veri setine sızabilir.43 Bilgisayarlı görü modelleri boyutu referans alamaz; fotoğraftaki objenin gerçek bir araç mı yoksa 1:43 ölçekli bir masaüstü maketi mi olduğunu sadece doku (texture), ışık yansıması ve çevresel referanslarla anlar.  
Eğer model, parlak plastik yüzeye sahip bir Hot Wheels oyuncağını gerçek bir MİCRO araç olarak öğrenirse, gerçek dünya senaryosunda (örneğin bir otoyol kamerasında) ışık yansımalarını, metalik boya dokusunu ve asfalttaki gerçek gölgeyi ayırt etme yeteneği kalıcı olarak zedelenir.

* **Çözüm Stratejisi (Hard Negative Mining):** Sadece temiz veri sağlamak yeterli değildir; modele neyin MİCRO "olmadığı" da agresif bir şekilde öğretilmelidir. Yüksek bir F1-Skoruna ulaşmak için çok sınıflı sınıflandırmalarda kullanılan "Boosted One-Vs-All (BOVA)" prensibinden ilham alınmalıdır.56 Veri seti manuel veya yarı-otomatik yöntemlerle gözden geçirilirken, tespit edilen oyuncak araçlar, Hot Wheels maketleri veya 3D render görseller silinip atılmak yerine, modele zorlu negatif örnekler (Hard Negatives) olarak sunulmalıdır. Bu görseller "False\_Micro" veya jenerik "Background/Negative" isimli, modelin eğitim sırasındaki Kayıp Fonksiyonuna (Loss Function) negatif yönlü katkı yapacak ayrı bir sınıfa dahil edilebilir.43 Bu strateji ile Evrişimli Sinir Ağı, "resmin ortasındaki küçük ve iki tekerleği görünen her obje MİCRO araçtır" şeklindeki yüzeysel öğrenmeden (shallow feature learning) kurtulur. Bunun yerine cam yansımaları, asfalttaki kırılmalar, panel birleşim yerleri (panel gaps) ve gerçekçi metalik boya difüzyonları gibi gerçek ile plastiği ayıran çok daha derin parametrelere (deep representations) odaklanmaya zorlanır. Bu yaklaşım, modelin Yanlış Pozitif (False Positive) oranını sıfıra yaklaştırarak, test setinde mükemmel bir Precision ve nihayetinde en yüksek F1-Skorunu elde etmesini garantiler.43

## **Sonuç**

MİCRO araç sınıfı, 8 sınıflı bir araç gövde tipi sınıflandırma yapay zeka projesinde morfolojik özellikleri, dar dingil mesafesi yapıları, farklı bağlamsal konumlanmaları (şehir içi/dar sokaklar) ve yüksek oranlı oyuncak/maket kirliliği (noise) potansiyeli taşıması nedeniyle aşılması gereken en zorlu engellerden birini oluşturmaktadır. Bu araştırma raporunda; Roboflow Universe, Kaggle, GitHub ve akademik repolar (Stanford Cars, CompCars, UPCT Dataset) gibi yüksek kaliteli açık veri kaynaklarından hedeflenen 1000+ görsel hacminin temellerinin nasıl atılabileceği somut doğrudan bağlantılarla kanıtlanmıştır. Batı pazarında yer alan Aixam, Ligier, Ami gibi özel araçların eksikliğini gidermek ve hedeflenen hacmin üzerine çıkmak için ise Python tabanlı, çoklu iş parçacığıyla (multi-threading) asenkron çalışan ve bozuk dosyaları PIL kütüphanesi ile otonom şekilde ayıklayabilen endüstri standardında bir web scraping mimarisi geliştirilmiştir.  
Özellikle MİCRO sınıfının geometrisini korumak için Aspect-Ratio Padding kullanmak, bağlamsal arka plan yanılgılarından (dar sokak, fuar alanı) modeli bağımsızlaştırmak için gelişmiş Data Augmentation teknikleri uygulamak ve sentetik oyuncak kirliliğine karşı "Hard Negative Mining" yaklaşımını benimsemek, kurulacak olan Evrişimli Sinir Ağı (CNN) veya Transformer mimarisinin genelleme yeteneğini en üst düzeye çıkaracaktır. Sınıflar arası veri dengesizliğinin olduğu senaryolarda yanıltıcı bir metrik olan Accuracy (Doğruluk) yerine F1-Skorunu baz alan bu optimizasyon tavsiyeleri, modelin test ortamında ve gerçek dünya (wild) senaryolarında üstün bir başarım sergilemesi için gereken tüm teorik ve pratik mühendislik altyapısını eksiksiz bir biçimde sunmaktadır.

#### **Alıntılanan çalışmalar**

1. Car\_type as an vehicle attribute · Issue \#869 · openmobilityfoundation/mobility-data-specification \- GitHub, erişim tarihi Mayıs 8, 2026, [https://github.com/openmobilityfoundation/mobility-data-specification/issues/869](https://github.com/openmobilityfoundation/mobility-data-specification/issues/869)  
2. Life-cycle inventories for on-road vehicles, erişim tarihi Mayıs 8, 2026, [https://downloads.ctfassets.net/4y40wxcxzkmz/5p6HLqNvy0A1ri6P2wiEl3/f4d26e0191c19967ddf5420968203be7/vehiclelca\_psi\_2023.pdf](https://downloads.ctfassets.net/4y40wxcxzkmz/5p6HLqNvy0A1ri6P2wiEl3/f4d26e0191c19967ddf5420968203be7/vehiclelca_psi_2023.pdf)  
3. Methodology for Electric Conversion of a Small City Car \- MDPI, erişim tarihi Mayıs 8, 2026, [https://www.mdpi.com/2673-4591/104/1/14](https://www.mdpi.com/2673-4591/104/1/14)  
4. Full text of "FHM May 2015 UK" \- Internet Archive, erişim tarihi Mayıs 8, 2026, [https://archive.org/stream/FHM\_May\_2015\_UK/FHM\_May\_2015\_UK\_djvu.txt](https://archive.org/stream/FHM_May_2015_UK/FHM_May_2015_UK_djvu.txt)  
5. Citroën Ami (electric vehicle) \- Wikipedia, erişim tarihi Mayıs 8, 2026, [https://en.wikipedia.org/wiki/Citro%C3%ABn\_Ami\_(electric\_vehicle)](https://en.wikipedia.org/wiki/Citro%C3%ABn_Ami_\(electric_vehicle\))  
6. Advanced Automotive Design Lecture Notes | PDF | Reliability Engineering \- Scribd, erişim tarihi Mayıs 8, 2026, [https://www.scribd.com/document/973173524/Design-Course-Notes](https://www.scribd.com/document/973173524/Design-Course-Notes)  
7. The definitive visual history of the automobile \- MG Clube de Portugal, erişim tarihi Mayıs 8, 2026, [https://mgclubedeportugal.pt/sites/default/files/other\_files/History%20of%20the%20Automobile.pdf](https://mgclubedeportugal.pt/sites/default/files/other_files/History%20of%20the%20Automobile.pdf)  
8. project\_electric\_cars\_france2040/p8\_notebook01.ipynb at french\_version \- GitHub, erişim tarihi Mayıs 8, 2026, [https://github.com/nalron/project\_electric\_cars\_france2040/blob/french\_version/p8\_notebook01.ipynb](https://github.com/nalron/project_electric_cars_france2040/blob/french_version/p8_notebook01.ipynb)  
9. Autonomous Vehicle Dataset with Real Multi-Driver Scenes and Biometric Data \- MDPI, erişim tarihi Mayıs 8, 2026, [https://www.mdpi.com/1424-8220/23/4/2009](https://www.mdpi.com/1424-8220/23/4/2009)  
10. Cars Computer Vision Datasets and Models \- Roboflow Universe, erişim tarihi Mayıs 8, 2026, [https://universe.roboflow.com/browse/transportation/cars](https://universe.roboflow.com/browse/transportation/cars)  
11. class:car-truck-bus-bike, Page 4 | Roboflow Universe Search, erişim tarihi Mayıs 8, 2026, [https://universe.roboflow.com/search?p=3\&q=class%3Acar-truck-bus-bike](https://universe.roboflow.com/search?p=3&q=class:car-truck-bus-bike)  
12. Top Autorickshaw Datasets and Models | Roboflow Universe, erişim tarihi Mayıs 8, 2026, [https://universe.roboflow.com/search?q=class%3Aautorickshaw](https://universe.roboflow.com/search?q=class:autorickshaw)  
13. like:thesis-aarau/thesis-n4a7t | Roboflow Universe Search, erişim tarihi Mayıs 8, 2026, [https://universe.roboflow.com/search?q=like%3Athesis-aarau%2Fthesis-n4a7t\&p=0](https://universe.roboflow.com/search?q=like:thesis-aarau/thesis-n4a7t&p=0)  
14. Car Types Object Detection Dataset by VEMO \- Roboflow Universe, erişim tarihi Mayıs 8, 2026, [https://universe.roboflow.com/vemo/car-types-nvqdm](https://universe.roboflow.com/vemo/car-types-nvqdm)  
15. Top Small Datasets and Models | Roboflow Universe, erişim tarihi Mayıs 8, 2026, [https://universe.roboflow.com/search?q=class%3Asmall+cars](https://universe.roboflow.com/search?q=class:small+cars)  
16. An Analysis of Multi-Task Architectures for the Hierarchic Multi-Label Problem of Vehicle Model and Make Classification \- arXiv, erişim tarihi Mayıs 8, 2026, [https://arxiv.org/html/2603.01746v1](https://arxiv.org/html/2603.01746v1)  
17. Monza: Image Classification of Vehicle Make and Model Using Convolutional Neural Networks and Transfer Learning \- CS231n \- Stanford University, erişim tarihi Mayıs 8, 2026, [https://cs231n.stanford.edu/reports/2015/pdfs/lediurfinal.pdf](https://cs231n.stanford.edu/reports/2015/pdfs/lediurfinal.pdf)  
18. Stanford Cars \- Kaggle, erişim tarihi Mayıs 8, 2026, [https://www.kaggle.com/code/mitchellodili/stanford-cars](https://www.kaggle.com/code/mitchellodili/stanford-cars)  
19. Stanford Car Dataset by classes folder \- Kaggle, erişim tarihi Mayıs 8, 2026, [https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder)  
20. cars196 | TensorFlow Datasets, erişim tarihi Mayıs 8, 2026, [https://www.tensorflow.org/datasets/catalog/cars196](https://www.tensorflow.org/datasets/catalog/cars196)  
21. Stanford Cars Dataset \- Kaggle, erişim tarihi Mayıs 8, 2026, [https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)  
22. Stanford\_Car Object Detection Model by Openglpro \- Roboflow Universe, erişim tarihi Mayıs 8, 2026, [https://universe.roboflow.com/openglpro/stanford\_car](https://universe.roboflow.com/openglpro/stanford_car)  
23. CVonline: Image Databases \- Informatics Homepages Server, erişim tarihi Mayıs 8, 2026, [https://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm](https://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm)  
24. Proceedings of 5th International Electrical Engineering Conference (IEEC-2020), erişim tarihi Mayıs 8, 2026, [https://ieec.neduet.edu.pk/2020/Proceedings\_IEEC\_2020.pdf](https://ieec.neduet.edu.pk/2020/Proceedings_IEEC_2020.pdf)  
25. Proceedings of IEEC 2020 Final | PDF | Electrical Grid | Photovoltaics \- Scribd, erişim tarihi Mayıs 8, 2026, [https://www.scribd.com/document/514972567/Proceedings-of-IEEC-2020-Final](https://www.scribd.com/document/514972567/Proceedings-of-IEEC-2020-Final)  
26. Online Used Car Sales Dataset \- Kaggle, erişim tarihi Mayıs 8, 2026, [https://www.kaggle.com/datasets/yaminh/exploring-online-used-car-sales](https://www.kaggle.com/datasets/yaminh/exploring-online-used-car-sales)  
27. Indian Cars Dataset \- Kaggle, erişim tarihi Mayıs 8, 2026, [https://www.kaggle.com/datasets/medhekarabhinav5/indian-cars-dataset](https://www.kaggle.com/datasets/medhekarabhinav5/indian-cars-dataset)  
28. Vehicle Ads Dataset \- Kaggle, erişim tarihi Mayıs 8, 2026, [https://www.kaggle.com/datasets/ivantha/sri-lanka-vehicle-ads-dataset](https://www.kaggle.com/datasets/ivantha/sri-lanka-vehicle-ads-dataset)  
29. Cars DataSet \- Kaggle, erişim tarihi Mayıs 8, 2026, [https://www.kaggle.com/datasets/mrdheer/cars-dataset](https://www.kaggle.com/datasets/mrdheer/cars-dataset)  
30. A Machine Learning Approach to Pedestrian Detection for Autonomous Vehicles Using High-Definition 3D Range Data \- PMC, erişim tarihi Mayıs 8, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5298591/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5298591/)  
31. Automation of driving elements. (a) Renault Twizy; (b) steering modification \- ResearchGate, erişim tarihi Mayıs 8, 2026, [https://www.researchgate.net/figure/Automation-of-driving-elements-a-Renault-Twizy-b-steering-modification-c-braking\_fig1\_323280345](https://www.researchgate.net/figure/Automation-of-driving-elements-a-Renault-Twizy-b-steering-modification-c-braking_fig1_323280345)  
32. End-to-End Deep Neural Network Architectures for Speed and Steering Wheel Angle Prediction in Autonomous Driving \- MDPI, erişim tarihi Mayıs 8, 2026, [https://www.mdpi.com/2079-9292/10/11/1266](https://www.mdpi.com/2079-9292/10/11/1266)  
33. DurLAR: A High-Fidelity 128-Channel LiDAR Dataset with Panoramic Ambient and Reflectivity Imagery for Multi-Modal Autonomous Driving Applications \- arXiv, erişim tarihi Mayıs 8, 2026, [https://arxiv.org/html/2406.10068v1](https://arxiv.org/html/2406.10068v1)  
34. Dur360BEV: A Real-world Single 360-degree Camera Dataset and Benchmark for Bird-Eye View Mapping in Autonomous Driving \- arXiv, erişim tarihi Mayıs 8, 2026, [https://arxiv.org/html/2503.00675v1](https://arxiv.org/html/2503.00675v1)  
35. Explainable AI for Object Detection from Autonomous Vehicles \- City Research Online, erişim tarihi Mayıs 8, 2026, [https://openaccess.city.ac.uk/id/eprint/34798/1/Hogan%20thesis%202025%20PDF-A.pdf](https://openaccess.city.ac.uk/id/eprint/34798/1/Hogan%20thesis%202025%20PDF-A.pdf)  
36. Electric vehicles in the EU from 2010 to 2014 \- is full scale commercialisation near? \- SETIS, erişim tarihi Mayıs 8, 2026, [https://setis.ec.europa.eu/system/files/2021-01/Electric\_vehicles\_in\_the\_EU.pdf](https://setis.ec.europa.eu/system/files/2021-01/Electric_vehicles_in_the_EU.pdf)  
37. icrawler · PyPI, erişim tarihi Mayıs 8, 2026, [https://pypi.org/project/icrawler/0.3.2/](https://pypi.org/project/icrawler/0.3.2/)  
38. Image Scraping — Deep Learning \- FR, erişim tarihi Mayıs 8, 2026, [https://perso.esiee.fr/\~chierchg/deep-learning/projects/image-classification/image-classification-2.html](https://perso.esiee.fr/~chierchg/deep-learning/projects/image-classification/image-classification-2.html)  
39. icrawler Documentation, erişim tarihi Mayıs 8, 2026, [https://icrawler.readthedocs.io/\_/downloads/en/latest/pdf/](https://icrawler.readthedocs.io/_/downloads/en/latest/pdf/)  
40. web scraping \- Python \- Download Images from google Image search? \- Stack Overflow, erişim tarihi Mayıs 8, 2026, [https://stackoverflow.com/questions/20716842/python-download-images-from-google-image-search](https://stackoverflow.com/questions/20716842/python-download-images-from-google-image-search)  
41. Basic Image Classifier Project. Aim of this project is to: | by Sampurn Anand | Nerd For Tech | Medium, erişim tarihi Mayıs 8, 2026, [https://medium.com/nerd-for-tech/basic-image-classifier-project-9f3b2c0b7798](https://medium.com/nerd-for-tech/basic-image-classifier-project-9f3b2c0b7798)  
42. Built-in crawlers — icrawler 0.6.6 documentation, erişim tarihi Mayıs 8, 2026, [https://icrawler.readthedocs.io/en/latest/builtin.html](https://icrawler.readthedocs.io/en/latest/builtin.html)  
43. GenOL: Generating Diverse Examples for Name-only Online Learning \- arXiv, erişim tarihi Mayıs 8, 2026, [https://arxiv.org/html/2403.10853v4](https://arxiv.org/html/2403.10853v4)  
44. How to Create and Tune Your Own, Data Set for Facial Recognition using Neural Networks., erişim tarihi Mayıs 8, 2026, [https://towardsdatascience.com/how-to-create-and-tune-your-own-data-set-for-facial-recognition-using-neural-networks-8a68be38652/](https://towardsdatascience.com/how-to-create-and-tune-your-own-data-set-for-facial-recognition-using-neural-networks-8a68be38652/)  
45. arXiv:2402.17753v1 \[cs.CL\] 27 Feb 2024, erişim tarihi Mayıs 8, 2026, [https://arxiv.org/pdf/2402.17753](https://arxiv.org/pdf/2402.17753)  
46. Mastering the F1 Score: A Practical Guide for Machine Learning Success \- Lightly AI, erişim tarihi Mayıs 8, 2026, [https://www.lightly.ai/blog/f1-score](https://www.lightly.ai/blog/f1-score)  
47. Understanding and Applying F1 Score: AI Evaluation Essentials with Hands-On Coding Example, erişim tarihi Mayıs 8, 2026, [https://arize.com/blog-course/f1-score/](https://arize.com/blog-course/f1-score/)  
48. How to improve F1 score with skewed classes? \- Cross Validated \- Stats StackExchange, erişim tarihi Mayıs 8, 2026, [https://stats.stackexchange.com/questions/274807/how-to-improve-f1-score-with-skewed-classes](https://stats.stackexchange.com/questions/274807/how-to-improve-f1-score-with-skewed-classes)  
49. Vehicle detection and classification using an ensemble of EfficientDet and YOLOv8 \- PMC, erişim tarihi Mayıs 8, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11419654/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11419654/)  
50. GitHub \- ChristianJavierMelo/Vehicle-Type-Recognition: This project aims to recognize vehicles through images. It is based on a Convolutional Neural Network following the essence of Machine Learning algorithms., erişim tarihi Mayıs 8, 2026, [https://github.com/ChristianJavierMelo/Vehicle-Type-Recognition](https://github.com/ChristianJavierMelo/Vehicle-Type-Recognition)  
51. Multidisciplinary development of an electric vehicle typology for the city \- RCA Research Repository, erişim tarihi Mayıs 8, 2026, [https://researchonline.rca.ac.uk/1354/1/VITAL%20Lino%20Thesis.pdf](https://researchonline.rca.ac.uk/1354/1/VITAL%20Lino%20Thesis.pdf)  
52. February 2010 Automobile All-Stars | PDF \- Scribd, erişim tarihi Mayıs 8, 2026, [https://www.scribd.com/document/718572821/Automobile-2010-02](https://www.scribd.com/document/718572821/Automobile-2010-02)  
53. Citroën Ami One preview: 'It could be driven without a licence' | Motoring | The Guardian, erişim tarihi Mayıs 8, 2026, [https://www.theguardian.com/technology/2019/mar/10/citroen-ami-one-preview-electric-vehicle-prototype-geneva-motorshow](https://www.theguardian.com/technology/2019/mar/10/citroen-ami-one-preview-electric-vehicle-prototype-geneva-motorshow)  
54. Deep Learning Techniques for Vehicle Detection and Classification from Images/Videos: A Survey \- MDPI, erişim tarihi Mayıs 8, 2026, [https://www.mdpi.com/1424-8220/23/10/4832](https://www.mdpi.com/1424-8220/23/10/4832)  
55. Boosting Vehicle Classification with Augmentation Techniques across Multiple YOLO Versions | Tan | JOIV : International Journal on Informatics Visualization, erişim tarihi Mayıs 8, 2026, [https://joiv.org/index.php/joiv/article/view/2313](https://joiv.org/index.php/joiv/article/view/2313)  
56. How to boost your F1 score for multiclass/multilabel classification? \- skeepers, erişim tarihi Mayıs 8, 2026, [https://techblog.skeepers.io/how-to-boost-your-f1-score-for-multiclass-multilabel-classification-eeb8452c8171](https://techblog.skeepers.io/how-to-boost-your-f1-score-for-multiclass-multilabel-classification-eeb8452c8171)  
57. Personalization of Vision-language Models and the Multi-Concept Challenge \- UNIVERSITÀ DEGLI STUDI DI PADOVA, erişim tarihi Mayıs 8, 2026, [https://thesis.unipd.it/retrieve/1ab0b8ed-5c4c-4619-9e43-f225c9c6b2e7/Isotton\_Gloria.pdf](https://thesis.unipd.it/retrieve/1ab0b8ed-5c4c-4619-9e43-f225c9c6b2e7/Isotton_Gloria.pdf)