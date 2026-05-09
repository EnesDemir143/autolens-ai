# Hugging Face Model Publishing Plan

## Goal

After each model run is fully finished and has usable artifacts, publish the completed model artifacts to the student's Hugging Face account as model repositories. This is separate from the Hugging Face Spaces demo: Spaces hosts the live Gradio app, while model repositories preserve trained artifacts, metrics, and model cards for presentation/report evidence.

## Recommended approach

Publish **models**, not the merged training dataset.

Recommended model repo strategy:

1. Create one model repo per finished candidate, for example:
   - `autolens-efficientnet-b2`
   - `autolens-mobilenetv4`
   - `autolens-resnet18`
   - `autolens-dinov3-vits16`
   - `autolens-dinov3-vits16-lora-merged`
2. Upload only finished, shareable artifacts:
   - `model.safetensors` plus metadata as the preferred clean weight artifact.
   - ONNX model artifact when available for deployment/inference.
   - Optional Lightning `.ckpt` only if there is a clear need for resume/reproducibility and the file does not contain problematic private metadata; by default keep `.ckpt` local.
   - `metadata.json` with class labels, preprocessing, image size, normalization, source run ID, metric summary, and artifact size.
   - `calibration.json` if temperature scaling was fitted.
   - `metrics.json` / model selection evidence if available.
   - Confusion matrix and plots if they are generated artifacts.
   - `README.md` model card.
3. Mark the final selected model clearly in the repo README and link the HF Spaces demo if deployed.


## Checkpoint format strategy

Use this artifact flow after a run is complete:

```text
Lightning .ckpt  ->  model.safetensors + metadata.json  ->  model.onnx
```

Rationale:

- Lightning `.ckpt` is best for resuming training because it may include model weights, optimizer/scheduler state, callback state, epoch/global step, and Lightning-specific metadata.
- `model.safetensors` is better for Hugging Face model repos because it stores tensor weights in a safer/shareable format and avoids bundling optimizer/callback/training state.
- ONNX is the deploy artifact for Gradio/HF Spaces CPU inference.
- ONNX export should prefer rebuilding the model from `model.safetensors` plus explicit metadata. Direct `.ckpt` export can remain a fallback when conversion is blocked.

Because safetensors contains only tensors, always save metadata beside it:

- architecture/model factory key
- source experiment config
- class labels and display labels
- image size, normalization, preprocessing version
- source checkpoint path/run ID
- metric summary
- base model/license notes
- calibration status/path if available

Do not assume a safetensors file alone is enough to reconstruct the model.

## Dataset publishing decision

Do **not** upload the merged image dataset to Hugging Face by default.

Reasoning:

- The project dataset is assembled from multiple public sources such as Kaggle/Hugging Face/public image datasets.
- Each source can have different license, redistribution, attribution, and platform terms.
- Even if the project can use the images for academic training, that does not automatically mean the merged image files can be redistributed as a new public dataset.
- Publishing the raw merged dataset could create avoidable license/redistribution risk.

Safer alternative:

- In model cards, document the dataset sources by name/link.
- Include source catalog CSV/summary if it contains URLs, source IDs, counts, and label mappings but does **not** redistribute restricted images.
- Include the class mapping, split policy, and cleaning methodology.
- If a source dataset already exists on Hugging Face, reference its Hub dataset ID in the model card metadata when appropriate.
- If a source is Kaggle-only, link the Kaggle dataset page in the model card text instead of uploading the images.

This gives report/presentation evidence without taking ownership of third-party dataset redistribution rights.

## Model card content

Each model repository should include a `README.md` model card. Hugging Face renders this file as the model page. The card should include:

- Model name and architecture.
- Task: 8-class car body type image classification.
- Intended use: AutoLens AI Yazlab 2 project demo and educational inference.
- Classes:
  - SUV
  - VAN
  - STATION WAGON
  - MICRO
  - OPEN WHEEL / F1
  - SEDAN
  - HATCHBACK
  - PICK UP
- Training dataset summary and source links.
- Preprocessing and image size.
- Training setup and important hyperparameters.
- Evaluation metrics: Accuracy, balanced accuracy, MCC, Precision, Recall, macro F1, weighted F1, per-class metrics.
- Normalized confusion matrix if available.
- Artifact size and CPU inference latency.
- Calibration status and temperature value if used.
- Limitations: source/domain bias, ambiguous body styles, no instructor final test leakage, CPU latency for heavier DINOv3 variants.
- Link to the Gradio/HF Spaces demo if available.

Example metadata header:

```yaml
---
library_name: onnxruntime
pipeline_tag: image-classification
tags:
  - computer-vision
  - image-classification
  - cars
  - onnx
  - gradio
license: other
---
```

Use `license: other` unless the exact model artifact license is clear. If the model is derived from a pretrained backbone such as DINOv3 or a timm model, mention the base model and its license/terms in the card.

## Upload methods

### Manual upload

1. Create a new Hugging Face model repository from the web UI.
2. Add/upload the final model folder contents.
3. Edit `README.md` as the model card.
4. Verify the files render correctly.

### CLI upload

After Hugging Face CLI authentication:

```bash
hf auth login
hf upload USERNAME/autolens-efficientnet-b2 artifacts/export/efficientnet_b2 . --repo-type model
```

Alternative older CLI form may be:

```bash
huggingface-cli upload USERNAME/autolens-efficientnet-b2 artifacts/export/efficientnet_b2 . --repo-type model
```

Use the command available in the local environment. The project Makefile can wrap this later with a target like:

```bash
make publish-model-hf REPO=USERNAME/autolens-efficientnet-b2 ARTIFACT_DIR=artifacts/export/efficientnet_b2
```

## What to publish now vs later

### Publish now / as soon as artifacts are complete

- Any model with a finished run, metrics, `model.safetensors`, metadata, and a stable artifact folder.
- Current best EfficientNet-B2 after safetensors conversion, ONNX export, and calibration.

### Publish later

- DINOv3 variants after their runs finish, safetensors conversion is verified, and ONNX export is verified.
- LoRA variant only after it is merged/exported into a single deployable artifact or clearly documented as an adapter-based checkpoint.
- Final selected model after final benchmark and calibration are complete.

## Acceptance checklist

Before publishing a model repo:

- [ ] No training dataset images are included.
- [ ] No instructor final test images are included.
- [ ] No `.env`, tokens, W&B secrets, Kaggle credentials, or private paths are included.
- [ ] Model card includes source dataset links/counts instead of redistributing restricted datasets.
- [ ] `model.safetensors` exists with metadata, or the reason it is unavailable is documented.
- [ ] ONNX artifact exists if the repo is intended for deployment.
- [ ] Artifact sizes are recorded.
- [ ] Metrics and confusion matrix are included if available.
- [ ] Calibration metadata is included if used.
- [ ] Gradio Space link is added if available.

## References

- Hugging Face Model Cards: https://huggingface.co/docs/hub/en/model-cards
- Hugging Face upload CLI: https://huggingface.co/docs/huggingface_hub/guides/upload
- Hugging Face Dataset Cards: https://huggingface.co/docs/hub/main/datasets-cards
- Hugging Face uploading datasets: https://huggingface.co/docs/hub/en/datasets-adding
