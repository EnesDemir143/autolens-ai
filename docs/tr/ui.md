# AutoLens AI — Arayüz Kullanımı

AutoLens AI iki farklı arayüzle çalışabilir. İkisi de aynı ONNX inference backend'ini ve `ONNXPredictor` sınıfını kullanır.

## Seçenek A — Gradio Demo

HF Spaces için en uygun basit arayüzdür.

```bash
make download-hf-model
make demo-gradio
```

Adres: `http://localhost:7860`

Özellikler:
- Görsel yükleme / clipboard paste
- Görsel önizleme
- ONNX Runtime inference
- Tahmin edilen sınıf, confidence, latency ve sınıf olasılıkları

## Seçenek B — FastAPI + React Web UI

Tam lokal web arayüzüdür.

```bash
make download-hf-model
make frontend-build
make demo-web
```

Adres: `http://localhost:8080`

Özellikler:
- 3 kolonlu modern desktop layout
- Görsel yükleme ve paste desteği
- Prediction card
- Softmax probability chart
- Model information panel
- Lokal safetensors varsa opsiyonel GradCAM endpoint

## Model İndirme

Model dosyaları GitHub'da tutulmaz. HuggingFace Hub'dan indirilir:

```bash
make download-hf-model
```

Farklı model seçmek için:

```bash
make download-hf-model MODEL=efficientnet-b2
make download-hf-model MODEL=resnet18
make download-hf-model MODEL=mobilenetv4
```

## HF Hub Linkleri

- [DINOv3 weighted](https://huggingface.co/morpeN1/autolens-dinov3-safe-weighted)
- [DINOv3 focal](https://huggingface.co/morpeN1/autolens-dinov3-safe-focal)
- [DINOv3 baseline](https://huggingface.co/morpeN1/autolens-dinov3-vits16)
- [EfficientNet-B2](https://huggingface.co/morpeN1/autolens-efficientnet-b2)
- [ResNet18](https://huggingface.co/morpeN1/autolens-resnet18)
- [MobileNetV4](https://huggingface.co/morpeN1/autolens-mobilenetv4)

Collection: [AutoLens Vehicle Body Classification](https://huggingface.co/collections/morpeN1/autolens-vehicle-body-classification)
