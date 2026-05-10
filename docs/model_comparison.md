# Model Karşılaştırma — Split Bazlı Baseline Sonuçları

**Tarih:** 2026-05-10

Bu dokümanda karşılaştırma split bazında ayrılmıştır. Ana seçim metriği **internal_test** üzerindeki F1-macro değeridir; train ve val sonuçları sadece öğrenme/genelleme davranışını görmek için eklenmiştir.

## Split Tanımları ve Kaynaklar

| Split | CSV | Sample | Bu dokümandaki rolü | Kaynak |
|---|---|---:|---|---|
| train | `artifacts/dataset/splits/train.csv` | 25285 | Eğitim performansı / overfit kontrolü | `checkpoints/*/metrics.csv` son train satırı |
| val | `artifacts/dataset/splits/val.csv` | 3157 | Model seçimi ve temperature fit split'i | `checkpoints/*/metrics.csv` son val satırı |
| internal_test | `artifacts/dataset/splits/internal_test.csv` | 3170 | **Asıl kıyas ve karar split'i** | `artifacts/export/*/internal_test_results.json` current safetensors inference |

> Not: Train satırlarında yalnızca accuracy/loss loglandı; train precision/recall/F1 yok. Bu yüzden F1 karşılaştırması train için yapılmıyor.
> Internal test loss current `internal_test_results.json` içinde saklanmadığı için ana internal-test tablosunda loss kullanılmıyor.

---

## Ana Karar Tablosu — Split: `internal_test`

Bu tablo **asıl kıyas tablosudur**. Tüm metrikler current export artefaktlarındaki `internal_test_results.json` confusion matrixlerinden hesaplandı.

| Rank | Model | Split | Params | Safetensors | Accuracy | Balanced Acc | MCC | Precision-macro | Recall-macro | F1-macro | F1-weighted |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **DINOv3 ViT-S/16** | internal_test | 21.6 M | 82.4 MB | **0.9438** | **0.9121** | **0.9314** | **0.9279** | **0.9121** | **0.9187** | **0.9442** |
| 2 | EfficientNet-B2 | internal_test | ~9 M | 29.7 MB | 0.9202 | 0.8923 | 0.9022 | 0.9056 | 0.8923 | 0.8982 | 0.9201 |
| 3 | ResNet18 | internal_test | ~11 M | 42.7 MB | 0.8968 | 0.8361 | 0.8734 | 0.8887 | 0.8361 | 0.8590 | 0.8963 |
| 4 | MobileNetV4-Conv-M | internal_test | ~9 M | 32.5 MB | 0.8905 | 0.8592 | 0.8665 | 0.8458 | 0.8592 | 0.8510 | 0.8912 |

**Karar:** DINOv3 ViT-S/16 internal_test F1-macro `0.9187` ile en iyi modeldir. EfficientNet-B2 ikinci sıradadır (`0.8982`).

---

## Train / Val / Internal Test Tek Bakış

Her satırda split açık yazılmıştır. Train metrikleri eğitim loop loglarıdır; val metrikleri validation loop loglarıdır; internal_test metrikleri current safetensors inference sonuçlarından hesaplanmıştır.

| Model | Split | Sample | Epoch | Accuracy | Loss | Precision-macro | Recall-macro | F1-macro | F1-weighted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv3 ViT-S/16 | train | 25285 | 11 | 0.9945 | 0.0192 | — | — | — | — |
| DINOv3 ViT-S/16 | val | 3157 | 11 | 0.9335 | 0.3581 | 0.9209 | 0.8953 | 0.9047 | 0.9331 |
| DINOv3 ViT-S/16 | internal_test | 3170 | — | 0.9438 | — | 0.9279 | 0.9121 | 0.9187 | 0.9442 |
| EfficientNet-B2 | train | 25285 | 16 | 0.9876 | 0.0428 | — | — | — | — |
| EfficientNet-B2 | val | 3157 | 16 | 0.9335 | 0.3194 | 0.8800 | 0.8996 | 0.8866 | 0.9337 |
| EfficientNet-B2 | internal_test | 3170 | — | 0.9202 | — | 0.9056 | 0.8923 | 0.8982 | 0.9201 |
| ResNet18 | train | 25285 | 18 | 0.9896 | 0.0263 | — | — | — | — |
| ResNet18 | val | 3157 | 18 | 0.8647 | 0.7452 | 0.8195 | 0.8093 | 0.8100 | 0.8666 |
| ResNet18 | internal_test | 3170 | — | 0.8968 | — | 0.8887 | 0.8361 | 0.8590 | 0.8963 |
| MobileNetV4-Conv-M | train | 25285 | 22 | 0.9879 | 0.0335 | — | — | — | — |
| MobileNetV4-Conv-M | val | 3157 | 22 | 0.8888 | 0.6263 | 0.8318 | 0.8412 | 0.8350 | 0.8894 |
| MobileNetV4-Conv-M | internal_test | 3170 | — | 0.8905 | — | 0.8458 | 0.8592 | 0.8510 | 0.8912 |

### Genelleme Özeti

| Model | Train Acc | Val Acc | Internal Test Acc | Train→Val Gap | Val→Internal Test Gap | Yorum |
|---|---:|---:|---:|---:|---:|---|
| DINOv3 ViT-S/16 | 0.9945 | 0.9335 | 0.9438 | 0.0610 | -0.0104 | En iyi internal-test genelleme |
| EfficientNet-B2 | 0.9876 | 0.9335 | 0.9202 | 0.0541 | 0.0133 | En güçlü CNN baseline; val yüksek ama test DINOv3 altında |
| ResNet18 | 0.9896 | 0.8647 | 0.8968 | 0.1249 | -0.0321 | CNN baseline |
| MobileNetV4-Conv-M | 0.9879 | 0.8888 | 0.8905 | 0.0991 | -0.0017 | CNN baseline |

---

## Per-Class Metrikler — Split: `internal_test`

Aşağıdaki per-class tablo sadece **internal_test** split’idir. Bu bölümde artık tüm modeller için aynı metrik tipi kullanılır: precision, recall ve F1.

| Model | Split | Class | Precision | Recall | F1 | Support |
|---|---|---|---:|---:|---:|---:|
| DINOv3 ViT-S/16 | internal_test | SUV | 0.9666 | 0.9060 | 0.9353 | 670 |
| DINOv3 ViT-S/16 | internal_test | VAN | 0.9759 | 0.9780 | 0.9769 | 455 |
| DINOv3 ViT-S/16 | internal_test | STATION WAGON | 0.9474 | 0.8182 | 0.8780 | 66 |
| DINOv3 ViT-S/16 | internal_test | MICRO | 0.9000 | 0.8182 | 0.8571 | 22 |
| DINOv3 ViT-S/16 | internal_test | OPEN WHEEL / F1 | 0.9983 | 1.0000 | 0.9991 | 586 |
| DINOv3 ViT-S/16 | internal_test | SEDAN | 0.9385 | 0.9486 | 0.9435 | 836 |
| DINOv3 ViT-S/16 | internal_test | HATCHBACK | 0.7945 | 0.8722 | 0.8315 | 266 |
| DINOv3 ViT-S/16 | internal_test | PICK UP | 0.9018 | 0.9554 | 0.9278 | 269 |
| EfficientNet-B2 | internal_test | SUV | 0.9213 | 0.9090 | 0.9151 | 670 |
| EfficientNet-B2 | internal_test | VAN | 0.9450 | 0.9824 | 0.9634 | 455 |
| EfficientNet-B2 | internal_test | STATION WAGON | 0.8033 | 0.7424 | 0.7717 | 66 |
| EfficientNet-B2 | internal_test | MICRO | 0.9545 | 0.9545 | 0.9545 | 22 |
| EfficientNet-B2 | internal_test | OPEN WHEEL / F1 | 0.9948 | 0.9846 | 0.9897 | 586 |
| EfficientNet-B2 | internal_test | SEDAN | 0.9023 | 0.9282 | 0.9151 | 836 |
| EfficientNet-B2 | internal_test | HATCHBACK | 0.7662 | 0.8008 | 0.7831 | 266 |
| EfficientNet-B2 | internal_test | PICK UP | 0.9574 | 0.8364 | 0.8929 | 269 |
| ResNet18 | internal_test | SUV | 0.8752 | 0.9104 | 0.8925 | 670 |
| ResNet18 | internal_test | VAN | 0.9630 | 0.9165 | 0.9392 | 455 |
| ResNet18 | internal_test | STATION WAGON | 0.8200 | 0.6212 | 0.7069 | 66 |
| ResNet18 | internal_test | MICRO | 0.9444 | 0.7727 | 0.8500 | 22 |
| ResNet18 | internal_test | OPEN WHEEL / F1 | 0.9636 | 0.9949 | 0.9790 | 586 |
| ResNet18 | internal_test | SEDAN | 0.8800 | 0.9031 | 0.8914 | 836 |
| ResNet18 | internal_test | HATCHBACK | 0.7306 | 0.7444 | 0.7374 | 266 |
| ResNet18 | internal_test | PICK UP | 0.9328 | 0.8253 | 0.8757 | 269 |
| MobileNetV4-Conv-M | internal_test | SUV | 0.9296 | 0.8284 | 0.8761 | 670 |
| MobileNetV4-Conv-M | internal_test | VAN | 0.9412 | 0.9495 | 0.9453 | 455 |
| MobileNetV4-Conv-M | internal_test | STATION WAGON | 0.5732 | 0.7121 | 0.6351 | 66 |
| MobileNetV4-Conv-M | internal_test | MICRO | 0.9048 | 0.8636 | 0.8837 | 22 |
| MobileNetV4-Conv-M | internal_test | OPEN WHEEL / F1 | 0.9682 | 0.9863 | 0.9772 | 586 |
| MobileNetV4-Conv-M | internal_test | SEDAN | 0.8811 | 0.9043 | 0.8926 | 836 |
| MobileNetV4-Conv-M | internal_test | HATCHBACK | 0.7050 | 0.7368 | 0.7206 | 266 |
| MobileNetV4-Conv-M | internal_test | PICK UP | 0.8633 | 0.8922 | 0.8775 | 269 |

---

## Zor Sınıflar — Split: `internal_test`, Metrik: F1

| Class | DINOv3 F1 | EfficientNet-B2 F1 | ResNet18 F1 | MobileNetV4 F1 | En iyi |
|---|---:|---:|---:|---:|---|
| STATION WAGON | 0.8780 | 0.7717 | 0.7069 | 0.6351 | DINOv3 |
| MICRO | 0.8571 | 0.9545 | 0.8500 | 0.8837 | EfficientNet-B2 |
| HATCHBACK | 0.8315 | 0.7831 | 0.7374 | 0.7206 | DINOv3 |
| PICK UP | 0.9278 | 0.8929 | 0.8757 | 0.8775 | DINOv3 |

---

## Model Boyutu — Scope: Export Artefaktı

Bu tablo performans split’i değildir; export artefakt boyutlarını ve 95 MB ödev limitini gösterir.

| Model | Scope | Safetensors | ONNX | DINOv3 Demo Latency | < 95 MB? |
|---|---|---:|---:|---:|:---:|
| DINOv3 ViT-S/16 | export_artifact | 82.4 MB | 82.6 MB | 39.6 ms avg / 44.2 ms single | ✅ |
| EfficientNet-B2 | export_artifact | 29.7 MB | 29.4 MB | previous demo smoke: 14.4 ms avg | ✅ |
| ResNet18 | export_artifact | 42.7 MB | 42.6 MB | not smoke-tested in final demo | ✅ |
| MobileNetV4-Conv-M | export_artifact | 32.5 MB | 32.1 MB | not smoke-tested in final demo | ✅ |

---

## Kalibrasyon — Fit Split: `val`, Eval Split: `internal_test`

Temperature scaling **yalnızca validation splitinde fit edildi**. Aşağıdaki before/after ECE ve NLL değerleri **internal_test** üzerinde ölçüldü.

| Model | Fit Split | Eval Split | Temperature | ECE before | ECE after | NLL before | NLL after |
|---|---|---|---:|---:|---:|---:|---:|
| DINOv3 ViT-S/16 | val | internal_test | 1.7211 | 0.0415 | **0.0136** | 0.2372 | **0.1784** |
| EfficientNet-B2 | val | internal_test | 1.7799 | 0.0393 | **0.0107** | 0.2745 | **0.2301** |
| ResNet18 | — | — | — | — | — | — | — |
| MobileNetV4-Conv-M | — | — | — | — | — | — | — |

> ResNet18 ve MobileNetV4 için calibration artefaktı yok; bu yüzden calibration satırları boş bırakıldı.

---

## Sonuç

**Kazanan model: DINOv3 ViT-S/16**

- Ana karar split’i `internal_test`; DINOv3 bu splitte en yüksek F1-macro değerine sahip: `0.9187`.
- DINOv3 aynı zamanda en yüksek internal-test accuracy değerine sahip: `0.9438`.
- EfficientNet-B2 en güçlü CNN baseline: internal-test F1-macro `0.8982`, accuracy `0.9202`.
- DINOv3 artefaktı `82.4 MB` safetensors ve `82.6 MB` ONNX ile 95 MB limitinin altında.
- Calibration kıyasına göre DINOv3 internal-test NLL `0.2372 → 0.1784`, ECE `0.0415 → 0.0136`; EfficientNet-B2 ECE açısından biraz daha düşük after değere sahip (`0.0107`) ama overall F1/accuracy DINOv3 daha iyi.

**Rapor/ Sunum için kullanılacak ana tablo:** `Ana Karar Tablosu — Split: internal_test`.
