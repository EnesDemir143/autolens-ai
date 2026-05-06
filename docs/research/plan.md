🔴 Öncelik 1: Kritik ve Zorunlu Katman (Core & Deep Learning)
Bu araçlar projenin iskeletidir. Modelin eğitilmesi, verinin işlenmesi ve cihaz (M2 Pro MPS) yönetimi için kesinlikle gereklidir.

torch & torchvision: Temel tensör işlemleri ve veri yükleme (DataLoader) altyapısı.

pytorch-lightning: Eğitim döngüsünü (training loop), MPS cihaz yönetimini ve EarlyStopping gibi proje isterlerini otomatize eden mühendislik katmanı.

timm (PyTorch Image Models): EfficientNet, MobileNetV3 gibi 95 MB altındaki modern ve optimize edilmiş önceden eğitilmiş (pre-trained) modelleri projeye dahil etmeni sağlar.

albumentations & opencv-python: Veri genellemesini (generalization) artırmak için en hızlı ve kapsamlı Data Augmentation (veri çoğaltma) araçları.

Pillow (PIL): Görsellerin standart okunma formatı.

🟠 Öncelik 2: Değerlendirme, Loglama ve Raporlama (Metrics & XAI)
Proje isterlerindeki "Accuracy, Precision, Recall, F1-Score" metriklerini hesaplamak ve IEEE raporunu görselleştirmek için kullanılacaklar.

torchmetrics: Eğitim sırasında F1-Score ve Accuracy değerlerini GPU/MPS üzerinde yavaşlamadan hesaplar.

wandb (Weights & Biases): Loss ve Accuracy grafiklerini canlı izlemek ve raporda kullanılacak görselleri elde etmek için.

scikit-learn & numpy & pandas: Eğitim sonrası detaylı "Classification Report" ve veri seti dağılım analizleri için.

matplotlib & seaborn: İstenen "Normalized Confusion Matrix (8x8 heatmap)" görselini yüksek çözünürlüklü çizebilmek için.

grad-cam: (Açıklanabilir Yapay Zeka) Modelin arabanın neresine bakarak tahmin yaptığını gösteren ısı haritaları üretir. Rapor ve sunum için muazzam bir eklentidir.

🟡 Öncelik 3: Optimizasyon ve Dağıtım (Deployment & UI)
Geliştirilen modelin limitlere (95 MB) uyması, hızlı çalışması ve arayüze bağlanması için gerekenler.

gradio: Sunum sırasında kullanılacak kullanıcı dostu, sürükle-bırak destekli web arayüzü.

safetensors: Model ağırlıklarını .pth formatına göre çok daha hızlı ve güvenli kaydetmek/yüklemek için.

onnx & onnxruntime: PyTorch modelini üretim (production) formatına çevirerek arayüzdeki tahmin hızını milisaniyelere düşürmek için.

(YENİ) onnxsim (ONNX Simplifier): Çevrilen ONNX modelinin grafiğini optimize ederek gereksiz işlemleri siler. 95 MB sınırını korumak ve modelin hızını daha da artırmak için kritik bir ektir.

torchinfo: Eğitime başlamadan önce modelin MB cinsinden boyutunu ve parametre sayısını kontrol etmeni sağlar.

🟢 Öncelik 4: MLOps ve Geliştirici Pratikleri (Code Quality & Config)
Kodun temiz, yönetilebilir ve hata payının minimum olmasını sağlayan modern yazılım mühendisliği araçları.

(YENİ) kaggle: Proje föyünde verilen referans veri setlerini (veya diğerlerini) terminal üzerinden tek satırla indirmek için resmi API.

(YENİ) tqdm: Terminalde verileri işlerken veya indirirken şık ilerleme çubukları (progress bar) gösterir.

optuna: En yüksek F1 skorunu elde etmek için öğrenme oranı (learning rate) gibi hiperparametreleri otomatik arayan optimizasyon kütüphanesi.

hydra-core & python-dotenv: Batch size, epoch gibi parametreleri temiz bir YAML dosyasından okumak ve API anahtarlarını güvende tutmak için.

ruff & mypy & pre-commit: Kodu PEP8 standartlarına göre anında formatlayan, gereksiz import'ları silen ve tip doğrulamasını statik olarak yapan kontrol mekanizmaları.

pytest: Modele tam eğitim vermeden önce mantık hatalarını tespit etmek için birim testleri (unit test) yazma aracı.

pyyaml,  rich/loguru, Hugging Face kullanacaksan huggingface_hub/datasets eklenicek birde.