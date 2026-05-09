# Phase 6: Hugging Face Spaces Demo Deploy and Model Publishing - Context
**Branch:** `feat/hugging-face-spaces-deploy`

**Gathered:** 2026-05-09
**Status:** Ready for planning
**Source:** User requested a report-before deployment phase after local Gradio demo is complete.

<domain>
## Phase Boundary

Hugging Face Spaces Demo Deploy and Model Publishing packages the already-working local Gradio + ONNX Runtime demo for a public/free CPU Basic Hugging Face Space, and prepares completed model artifacts for upload to the student Hugging Face account as model repositories. This phase does not add Docker, a production API, training on server, new model logic, or raw dataset redistribution. It only prepares deploy/model-publishing files, documentation, and optional CLI/manual upload paths.

</domain>

<decisions>
## Implementation Decisions

### D-01
- Use Hugging Face Spaces with the Gradio SDK, not Docker, as the default deploy target.

### D-02
- Run inference only on Spaces CPU Basic. Training, tuning, and model selection remain local/MPS/W&B workflows.

### D-03
- Deploy uses the same active artifact config from Phase 4/5: ONNX model path, metadata, class mapping, preprocessing, and calibration temperature.

### D-04
- Keep deploy packaging small: `app.py`, `requirements.txt`, active config, ONNX artifact(s), metadata/calibration JSON, and minimal source package code needed for inference/UI.

### D-05
- Provide both manual instructions and CLI-friendly commands when possible. If Hugging Face auth/token or repo creation is missing, document exact manual steps instead of blocking the project.

### D-06
- Docker is explicitly out of scope unless Gradio SDK deploy fails for a documented reason.

### D-07
- Publish finished model artifacts to Hugging Face model repos when useful for evidence/portfolio, but do not upload the merged raw dataset by default because source licenses and redistribution rights may differ.

### D-08
- Model cards should document dataset source links, counts, class mapping, training/evaluation metrics, calibration, limitations, and HF Spaces demo link when available.

### the agent's Discretion
- Exact deploy directory name may be chosen during execution, but prefer something clear like `deploy/hf_space/`.
- Keep the deploy artifact explainable and small; avoid copying training-only code or datasets into the Space.

</decisions>

<canonical_refs>
## Canonical References

- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/phases/04-model-comparison-and-selection/05-05-current-checkpoint-export-calibration-PLAN.md`
- `.planning/phases/05-gradio-demo-interface/05-VALIDATION.md`
- `docs/hf_model_publishing.md`
- Hugging Face Spaces Gradio docs and dependency docs when implementing deploy steps.
- `.agents/skills/andrej-karpathy-skills/skills/karpathy-guidelines/SKILL.md`

</canonical_refs>

<specifics>
## Specific Ideas

- Local demo must pass before this phase starts.
- Target deploy stack: Hugging Face Spaces + Gradio SDK + ONNX Runtime CPU.
- Required files should include `app.py`, `requirements.txt`, README/Space metadata, model config, ONNX model, calibration metadata, and class labels.
- Add Makefile targets such as `make package-hf-space` and optionally `make deploy-hf-space SPACE=owner/name` if CLI auth is available.
- Add model publishing targets/docs such as `make publish-model-hf REPO=owner/model ARTIFACT_DIR=artifacts/export/...` if CLI auth is available.
- Publish model artifacts and model cards; do not publish merged raw datasets unless licenses are individually cleared.
- Provide fallback manual steps: create Space, select Gradio SDK, upload/push files, watch build logs, test link.

</specifics>

<deferred>
## Deferred Ideas

- Docker Spaces, FastAPI, paid GPU deployment, custom domains, and production monitoring are deferred/out of scope.
- Report generation remains Phase 7 after deploy evidence/link is available.

</deferred>

---
*Phase: 06-hugging-face-spaces-deploy*
*Context gathered: 2026-05-09 via GSD planning update*
