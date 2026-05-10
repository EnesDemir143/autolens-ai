# Final Model Rationale — DINOv3 ViT-S/16 Non-LoRA

**Date:** 2026-05-10  
**Decision:** Use **DINOv3 ViT-S/16 non-LoRA** as the final AutoLens AI model candidate.  
**Primary decision split:** `internal_test` (`artifacts/dataset/splits/internal_test.csv`, 3170 samples).  
**Primary metric:** F1-macro, then Accuracy, then deployability under the 95 MB artifact limit.

---

## Decision Summary

DINOv3 ViT-S/16 non-LoRA is selected because it is the strongest completed and deployable model across the evidence that matters for the assignment:

1. It has the best **internal_test F1-macro** among all exported candidates.
2. It has the best **internal_test accuracy** among all exported candidates.
3. It is under the **95 MB model artifact limit**.
4. It has a complete deployment package: `model.safetensors`, `model.onnx`, `metadata.json`, `calibration.json`, size evidence, internal-test evidence, and a Hugging Face model card.
5. The LoRA variant did not beat the non-LoRA model on validation evidence, and the completed LoRA internal-test run was far below the non-LoRA model.

The final choice is therefore not simply “DINOv3”; it is specifically **DINOv3 ViT-S/16 without LoRA adapters**.

---

## Non-LoRA vs LoRA Evidence

| Variant | Run | Split | Accuracy | F1-macro | F1-weighted | Loss | Status |
|---|---|---|---:|---:|---:|---:|---|
| **DINOv3 ViT-S/16 non-LoRA** | `baseline_0_dinov3_vits16_20260509_235120` | **internal_test** | **0.9438** | **0.9187** | **0.9442** | 0.2415 | Exported, calibrated, HF-published |
| DINOv3 ViT-S/16 non-LoRA | `baseline_0_dinov3_vits16_20260509_235120` | best val | **0.9474** | **0.9238** | **0.9477** | 0.2714 | Best validation checkpoint evidence |
| DINOv3 ViT-S/16 LoRA | `baseline_0_dinov3_vits16_lora_20260510_073524` | best val | 0.9176 | 0.8926 | 0.9187 | 0.2467 | Did not beat non-LoRA; no internal-test/export evidence in current artifacts |
| DINOv3 ViT-S/16 LoRA | `baseline_0_dinov3_vits16_lora_20260509_225159` | internal_test | 0.7820 | 0.6316 | 0.7797 | 0.7305 | Inferior completed test run |

### Interpretation

- The **best non-LoRA validation F1-macro** is `0.9238`; the later LoRA run reaches only `0.8926` on best validation F1-macro.
- The **completed LoRA internal-test run** is much worse (`0.6316` F1-macro), so it cannot support choosing LoRA.
- The later LoRA run improved validation performance, but it still did **not** surpass non-LoRA validation performance and does not currently have the same final evidence chain: internal-test evaluation, export, calibration, size check, and HF model card.
- Since the non-LoRA model already satisfies performance and deployability constraints, LoRA adds complexity without a measured benefit.

---

## Why LoRA Was Not Selected

LoRA is useful when full fine-tuning is too expensive or when adapter-based updates improve performance under resource constraints. In this project, however, the measured evidence does not justify choosing the LoRA path.

### 1. LoRA did not outperform non-LoRA

The best LoRA validation F1-macro (`0.8926`) is below the non-LoRA validation F1-macro (`0.9238`). The available LoRA internal-test result is also far below the selected model.

### 2. LoRA has weaker final evidence

The selected non-LoRA model has a full final-artifact chain:

- `artifacts/export/dinov3_vits16_latest/model.safetensors`
- `artifacts/export/dinov3_vits16_latest/model.onnx`
- `artifacts/export/dinov3_vits16_latest/metadata.json`
- `artifacts/export/dinov3_vits16_latest/internal_test_results.json`
- `artifacts/export/dinov3_vits16_latest/calibration.json`
- `artifacts/export/dinov3_vits16_latest/size_check.json`
- Hugging Face model card and calibration evidence

The LoRA variant does not currently have equivalent final export/calibration/HF evidence.

### 3. LoRA increases deployment complexity

A LoRA final model would require either:

- serving a base model plus adapter weights, or
- merging adapter weights into a single deployable artifact and re-running export/calibration/size checks.

That extra complexity is only worth it if LoRA clearly improves F1-macro, accuracy, or model size. Current evidence does not show that.

### 4. Assignment constraints favor the already-complete non-LoRA artifact

The project deadline and artifact-size requirement favor a model that is already exported, calibrated, and publishable. The non-LoRA DINOv3 artifact is under 95 MB and already meets the final evidence requirements.

---

## Why DINOv3 Non-LoRA Beats the CNN Baselines

The broader model comparison in `docs/model_comparison.md` shows that DINOv3 non-LoRA is also the strongest overall model among the completed candidates.

| Model | Split | Accuracy | F1-macro | F1-weighted | Safetensors |
|---|---|---:|---:|---:|---:|
| **DINOv3 ViT-S/16 non-LoRA** | internal_test | **0.9438** | **0.9187** | **0.9442** | 82.4 MB |
| EfficientNet-B2 | internal_test | 0.9202 | 0.8982 | 0.9201 | 29.7 MB |
| ResNet18 | internal_test | 0.8968 | 0.8590 | 0.8963 | 42.7 MB |
| MobileNetV4-Conv-M | internal_test | 0.8905 | 0.8510 | 0.8912 | 32.5 MB |

EfficientNet-B2 has a strong MICRO-class result, but MICRO has only 22 internal-test samples. That isolated result is not enough to override DINOv3’s better overall F1-macro, accuracy, and difficult-class balance.

---

## Report-Ready Rationale

DINOv3 ViT-S/16 without LoRA was selected as the final model because it achieved the highest internal-test macro F1 and accuracy among all completed candidates while remaining below the 95 MB artifact limit. The LoRA variant was evaluated as an optional comparison path, but it did not outperform the non-LoRA model on validation metrics, and the completed LoRA internal-test run was substantially worse. Because LoRA would also require additional adapter merge/export/calibration steps, it introduces deployment complexity without measured benefit. Therefore, the non-LoRA DINOv3 artifact is the most reliable final choice for the project’s performance, deadline, and deployment constraints.

---

## Final Decision

**Selected final model:** `DINOv3 ViT-S/16 non-LoRA`  
**Selected run:** `baseline_0_dinov3_vits16_20260509_235120`  
**Selected checkpoint:** `checkpoints/baseline_0_dinov3_vits16_20260509_235120/best-09-0.9238.ckpt`  
**Final export directory:** `artifacts/export/dinov3_vits16_latest/`  
**Primary evidence file:** `docs/model_comparison.md`
