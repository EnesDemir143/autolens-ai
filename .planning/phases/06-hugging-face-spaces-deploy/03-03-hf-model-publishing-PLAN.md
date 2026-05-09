---
phase: 6
phase_name: Hugging Face Spaces Demo Deploy and Model Publishing
branch: feat/hugging-face-spaces-deploy
plan_id: 06-03
objective: Publish completed model artifacts to Hugging Face model repositories without redistributing raw datasets
wave: 2
depends_on: ['06-01']
requirements_addressed: ['PUBLISH-01', 'PUBLISH-02', 'PUBLISH-03']
files_modified: ['docs/hf_model_publishing.md', 'Makefile', 'artifacts/export/', 'deploy/hf_models/']
autonomous: true
---

# Plan 03 — Publish completed model artifacts to Hugging Face model repos

<objective>
Prepare finished model artifact folders and model cards so completed AutoLens AI candidates can be uploaded to the student's Hugging Face account as model repositories. Avoid uploading the merged raw image dataset unless every source license and redistribution term is cleared.
</objective>

<must_haves>
- Covers requirement IDs: PUBLISH-01, PUBLISH-02, PUBLISH-03.
- Publish `model.safetensors`, ONNX artifacts when available, metadata, metrics, calibration files, and model cards. Keep Lightning `.ckpt` local by default unless explicitly needed for reproducibility.
- Do not publish raw merged dataset images by default.
</must_haves>

<tasks>

<task id="P6-03-T1" type="execute">
  <title>Prepare model publishing guide and artifact checklist</title>
  <read_first>
    - `docs/hf_model_publishing.md`
    - `.planning/phases/06-hugging-face-spaces-deploy/06-VALIDATION.md`
  </read_first>
  <action>
    Ensure the guide explains the artifact flow `.ckpt -> model.safetensors + metadata -> ONNX`, which artifacts to upload, how to write model cards, how to document dataset sources by links/counts, and why the merged raw dataset is not uploaded by default.
  </action>
  <acceptance_criteria>
    - Guide exists under `docs/`.
    - Dataset license/redistribution caveat is explicit.
    - Finished model artifact checklist is explicit and prefers safetensors+metadata over raw Lightning checkpoints for HF sharing.
  </acceptance_criteria>
</task>

<task id="P6-03-T2" type="execute">
  <title>Add optional HF model upload command path</title>
  <action>
    Add or document a Makefile/CLI path for uploading a chosen artifact directory to a Hugging Face model repo, using variables such as `REPO=owner/model` and `ARTIFACT_DIR=artifacts/export/...`; the artifact directory should contain `model.safetensors`, metadata, ONNX if available, calibration, metrics, plots, and README. If Hugging Face auth is missing, document `hf auth login` and manual web upload fallback.
  </action>
  <acceptance_criteria>
    - Upload command path is discoverable from docs or `make help`.
    - Command does not upload datasets, secrets, or instructor final test images.
    - Missing auth is documented as a user action rather than a code blocker.
  </acceptance_criteria>
</task>

<task id="P6-03-T3" type="execute">
  <title>Record published model evidence</title>
  <action>
    For each uploaded model repo, record the repo URL, artifact version, source run ID, metrics summary, and whether it is a candidate or final selected model. Feed this evidence into Phase 7 report inputs.
  </action>
  <acceptance_criteria>
    - Published model repo links or pending/manual upload status are recorded.
    - Phase 7 can cite the model repo(s) without guessing.
  </acceptance_criteria>
</task>

</tasks>

<verification>
Use `06-VALIDATION.md`. Check no raw dataset images, secrets, `.env`, or final instructor test files are included in publish folders.
</verification>
