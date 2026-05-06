# Phase 4 Research: Main DINOv3 Model and Selection
**Branch:** `feat/model-comparison-and-selection`

    ## Research Question

    What needs to be known to plan Phase 4 well?

    ## Comparison Research Notes

DINOv3 model `facebook/dinov3-vits16-pretrain-lvd1689m` is the intended main model path. It is a gated Hugging Face model with about 21.6M parameters and transformers support, so planning must include early access validation plus fallback if access/size/latency blocks deployment. MobileNetV4, EfficientNet-B2, and ResNet are baseline/comparison models. All models must be scored by macro F1, weighted F1, per-class metrics, artifact size, and latency.


    ## Validation Architecture

    Phase 4 validation checks:
1. metric suite computes required metrics.
2. plots are generated to artifacts directory.
3. DINOv3 main-path access/fallback is documented.
4. benchmark table contains DINOv3 vs baseline F1/size/latency.
5. selected final model is under 95 MB or mitigation documented.
