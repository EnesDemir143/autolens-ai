# AutoLens AI — Model Karşılaştırması

Bu doküman, AutoLens AI projesinde eğitilen modellerin final karşılaştırmasını özetler. Tüm modeller aynı **internal_test** split üzerinde değerlendirilmiştir. Ana seçim metriği **macro F1-score** olarak belirlenmiştir.

## Final Sıralama

| Rank | Model | Mimari | Parametre | Safetensors | Accuracy | F1-macro | F1-weighted |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | **DINOv3 ViT-S/16 weighted** | Vision Transformer | 21.6M | 82.4 MB | **0.9438** | **0.9187** | **0.9442** |
| 2 | EfficientNet-B2 | CNN | ~9M | 29.7 MB | 0.9202 | 0.8982 | 0.9201 |
| 3 | ResNet18 | CNN | ~11M | 42.7 MB | 0.8968 | 0.8590 | 0.8963 |
| 4 | MobileNetV4-Conv-M | CNN | ~9M | 32.5 MB | 0.8905 | 0.8510 | 0.8912 |

## Seçilen Model: DINOv3 ViT-S/16 Weighted

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUV | 0.9666 | 0.9060 | 0.9353 | 670 |
| VAN | 0.9759 | 0.9780 | 0.9769 | 455 |
| STATION WAGON | 0.9474 | 0.8182 | 0.8780 | 66 |
| MICRO | 0.9000 | 0.8182 | 0.8571 | 22 |
| OPEN WHEEL / F1 | 0.9983 | 1.0000 | 0.9991 | 586 |
| SEDAN | 0.9385 | 0.9486 | 0.9435 | 836 |
| HATCHBACK | 0.7945 | 0.8722 | 0.8315 | 266 |
| PICK UP | 0.9018 | 0.9554 | 0.9278 | 269 |

## Analiz

DINOv3 ViT-S/16 weighted modeli şu nedenlerle seçildi:

1. Internal test üzerinde en yüksek macro F1: **0.9187**
2. En yüksek accuracy: **0.9438**
3. 95 MB model boyutu şartını sağlıyor: **82.6 MB ONNX**
4. Az örnekli sınıflarda güçlü performans veriyor: MICRO, STATION WAGON
5. Validation → test genellemesi stabil

## Zor Sınıflar

- **STATION WAGON:** Support düşük (66), görsel olarak sedan/hatchback ile karışabiliyor.
- **MICRO:** En az örneğe sahip sınıf (22 test örneği), weighted loss fayda sağladı.
- **HATCHBACK:** SUV/Sedan ile görsel benzerlikten dolayı hata alabiliyor.

## Model Boyutları

| Model | Safetensors | ONNX | <95 MB? |
|---|---:|---:|:---:|
| DINOv3 ViT-S/16 | 82.4 MB | 82.6 MB | ✅ |
| EfficientNet-B2 | 29.7 MB | 29.4 MB | ✅ |
| ResNet18 | 42.7 MB | 42.6 MB | ✅ |
| MobileNetV4 | 32.5 MB | 32.1 MB | ✅ |

## Sonuç

Final deploy modeli **DINOv3 ViT-S/16 weighted** olarak seçilmiştir. EfficientNet-B2 en güçlü CNN baseline olarak ikinci sıradadır.