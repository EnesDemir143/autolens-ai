# Phase 4 Research: Model Comparison and Selection

    ## Research Question

    What needs to be known to plan Phase 4 well?

    ## Comparison Research Notes

DINOv3 model `facebook/dinov3-vits16-pretrain-lvd1689m` is a gated Hugging Face model with about 21.6M parameters and transformers support. Planning must include access validation and fallback. All models must be scored by macro F1, weighted F1, per-class metrics, artifact size, and latency.


    ## Validation Architecture

    Phase 4 validation checks:
1. metric suite computes required metrics.
2. plots are generated to artifacts directory.
3. DINOv3 access/fallback is documented.
4. benchmark table contains F1/size/latency.
5. selected final model is under 95 MB or mitigation documented.
