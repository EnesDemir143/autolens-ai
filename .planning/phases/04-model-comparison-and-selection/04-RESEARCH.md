# Phase 4 Research: Main DINOv3 Model and Selection
**Branch:** `feat/model-comparison-and-selection`

    ## Research Question

    What needs to be known to plan Phase 4 well?

    ## Comparison Research Notes

DINOv3 model `facebook/dinov3-vits16-pretrain-lvd1689m` is the intended main model path. It is a gated Hugging Face model with about 21.6M parameters and transformers support, so planning must include early access validation plus fallback if access/size/latency blocks deployment. The overall comparison has four model families: MobileNetV4 Conv Medium, EfficientNet-B2, ResNet18, and DINOv3. DINOv3 must be evaluated without LoRA first; a LoRA fine-tuning variant is a separate optional comparison after non-LoRA works. All models must be scored by macro F1, weighted F1, balanced accuracy, MCC, per-class metrics, artifact size, and latency. Optional supporting diagnostics from the imbalance notebooks include per-class specificity, ECE, Brier score, Cohen's kappa, and reliability curves; these are helpful for confidence interpretation but are not required core assignment metrics.


    ## Validation Architecture

    Phase 4 validation checks:
1. metric suite computes required metrics.
2. plots are generated to artifacts directory.
3. DINOv3 main-path access/fallback is documented.
4. benchmark table contains four model-family results with DINOv3 non-LoRA, optional DINOv3 LoRA, baseline F1/balanced-accuracy/MCC, size, and latency.
5. selected final model is under 95 MB or mitigation documented.
