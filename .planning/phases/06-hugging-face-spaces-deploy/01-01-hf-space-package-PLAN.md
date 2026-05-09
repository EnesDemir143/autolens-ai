---
phase: 6
phase_name: Hugging Face Spaces Demo Deploy
branch: feat/hugging-face-spaces-deploy
plan_id: 06-01
objective: Package local Gradio ONNX demo for Hugging Face Spaces Gradio SDK
wave: 1
depends_on: ['05-04']
requirements_addressed: ['DEPLOY-01', 'DEPLOY-02', 'DEPLOY-03']
files_modified: ['deploy/hf_space/', 'Makefile', 'docs/deployment_runbook.md']
autonomous: true
---

# Plan 01 — Package local Gradio ONNX demo for Hugging Face Spaces

<objective>
Create a minimal Hugging Face Spaces deploy package that runs the existing Gradio + ONNX Runtime inference path on CPU Basic without Docker.
</objective>

<must_haves>
- Covers requirement IDs: DEPLOY-01, DEPLOY-02, DEPLOY-03.
- Local demo must be working before packaging.
- Do not copy training data, raw datasets, instructor final test images, secrets, or unnecessary checkpoints.
</must_haves>

<tasks>

<task id="P6-01-T1" type="execute">
  <title>Define deploy package contents</title>
  <read_first>
    - `.planning/phases/06-hugging-face-spaces-deploy/06-CONTEXT.md`
    - `.planning/phases/06-hugging-face-spaces-deploy/06-VALIDATION.md`
    - `.planning/phases/05-gradio-demo-interface/05-VALIDATION.md`
  </read_first>
  <action>
    Define the minimal deploy directory contents: Space `app.py`, `requirements.txt`, active model config, ONNX model, metadata/calibration JSON, class labels, and only the source files required for inference/UI.
  </action>
  <acceptance_criteria>
    - Deploy package structure is documented.
    - Training-only files and datasets are explicitly excluded.
  </acceptance_criteria>
</task>

<task id="P6-01-T2" type="execute">
  <title>Add packaging command</title>
  <action>
    Add a Makefile target or script that copies the active demo artifact into the HF Space package directory and validates required files exist.
  </action>
  <acceptance_criteria>
    - `make help` documents the package command and required variables.
    - Packaging command fails clearly if active ONNX/config files are missing.
  </acceptance_criteria>
</task>

</tasks>

<verification>
Use `06-VALIDATION.md`; run local package smoke checks where possible.
</verification>
