---
phase: 4
phase_name: Export, Calibration, Overnight Model Selection
branch: feat/model-comparison-and-selection
plan_id: 04-05
objective: Convert current EfficientNet-B2 Lightning checkpoint to safetensors, export ONNX, check size, calibrate, and produce swappable inference metadata
wave: 0
depends_on: []
requirements_addressed: ['OPT-01', 'OPT-02', 'CAL-01', 'CAL-02', 'EVAL-07', 'OPS-02', 'PUBLISH-01']
files_modified: ['Makefile', 'src/autolens_ai/inference/', 'src/autolens_ai/evaluation/', 'scripts/', 'artifacts/export/', 'docs/model_selection.md', 'docs/deployment_runbook.md', 'docs/hf_model_publishing.md']
autonomous: true
---

# Plan 05 — Convert checkpoint, export ONNX, calibrate confidence, and prepare demo artifact

<objective>
Convert the current best EfficientNet-B2 Lightning checkpoint into a clean `model.safetensors` artifact plus metadata, export ONNX from that deploy/share artifact when possible, verify the 95 MB limit, fit validation-only temperature scaling, and save explicit metadata that Phase 5 can load without knowing which model family ultimately wins.
</objective>

<must_haves>
- Covers requirement IDs: OPT-01, OPT-02, CAL-01, CAL-02, EVAL-07, OPS-02, PUBLISH-01.
- This plan is wave 0 because it unblocks the working demo before overnight experiments finish.
- It must not modify training results or consume internal_test/instructor final test data.
- It must add Makefile targets/runbook entries that encode the correct post-training order so a chosen run/checkpoint can be deployed without manual command reconstruction.
- Lightning `.ckpt` remains the training-resume artifact; `model.safetensors` plus metadata becomes the preferred share/export source for Hugging Face model repos and ONNX export.
</must_haves>

<threat_model>
Keep local credentials and dataset/API tokens out of git. Do not upload private data or final instructor test data into artifacts. Validation predictions may be used for temperature scaling; internal_test is reserved for the final evaluation pass.
</threat_model>

<tasks>

<task id="P4-05-T1" type="execute">
  <title>Locate current EfficientNet-B2 best checkpoint and metadata</title>
  <read_first>
    - `.planning/phases/04-model-comparison-and-selection/04-CONTEXT.md`
    - `.planning/phases/04-model-comparison-and-selection/04-VALIDATION.md`
    - `docs/baseline_training_runbook.md`
    - `docs/phase_reports/phase_03_baseline_runs.md`
  </read_first>
  <action>
    Record the current best EfficientNet-B2 checkpoint path, source config, class mapping, image size, normalization, and validation split manifest. If the checkpoint is missing, document the exact expected path and the command needed to regenerate it.
  </action>
  <acceptance_criteria>
    - Checkpoint path or exact blocker is documented.
    - Class mapping and preprocessing metadata are identified.
  </acceptance_criteria>
</task>

<task id="P4-05-T1B" type="execute">
  <title>Convert Lightning checkpoint to safetensors with explicit metadata</title>
  <action>
    Add the smallest conversion path from Lightning `.ckpt` to `model.safetensors`. Save only model weights in safetensors and save non-tensor context separately: architecture/model factory name, config path, class mapping, preprocessing/image size/normalization, source checkpoint path, training run ID, git state, metrics summary if available, and base model/license notes. Keep the original `.ckpt` for resume/debugging; do not treat safetensors as a full Lightning training checkpoint.
  </action>
  <acceptance_criteria>
    - `model.safetensors` exists or exact conversion blocker is documented.
    - Metadata JSON/YAML exists beside `model.safetensors` and is sufficient to reconstruct the model for inference/export.
    - The conversion does not include optimizer state, callbacks, datamodule internals, secrets, or dataset images.
    - Original Lightning `.ckpt` remains available for training resume if needed.
  </acceptance_criteria>
</task>

<task id="P4-05-T2" type="execute">
  <title>Export safetensors artifact to ONNX and check artifact size</title>
  <action>
    Add the smallest export path needed to rebuild the model from `model.safetensors` plus metadata and produce ONNX. Keep direct `.ckpt` export as a fallback only if safetensors reconstruction is blocked. Run ONNX Runtime smoke inference, optionally simplify with onnxsim, and record safetensors/raw-ONNX/simplified-ONNX artifact sizes against the 95 MB limit.
  </action>
  <acceptance_criteria>
    - ONNX artifact exists from safetensors+metadata or exact export blocker/fallback is documented.
    - ONNX Runtime smoke check is recorded.
    - Size evidence is recorded; artifact is under 95 MB or mitigation is documented.
  </acceptance_criteria>
</task>

<task id="P4-05-T3" type="execute">
  <title>Fit validation-only temperature scaling</title>
  <action>
    Fit temperature scaling using validation predictions only. Save calibration temperature and before/after confidence diagnostics. Do not use internal_test or instructor final test data.
  </action>
  <acceptance_criteria>
    - Calibration metadata exists or exact blocker is documented.
    - Evidence includes before/after NLL/ECE or equivalent diagnostics where possible.
    - No internal_test or instructor final test leakage occurs.
  </acceptance_criteria>
</task>

<task id="P4-05-T4" type="execute">
  <title>Write swappable inference artifact contract for Phase 5</title>
  <action>
    Document the artifact contract consumed by Gradio: model path, backend type, class mapping, preprocessing metadata, calibration temperature, and display labels. Make replacement of the final winner a config/artifact change rather than a UI rewrite.
  </action>
  <acceptance_criteria>
    - Phase 5 can load the current artifact through explicit config/metadata.
    - Final winner replacement instructions are documented.
  </acceptance_criteria>
</task>

<task id="P4-05-T5" type="execute">
  <title>Add Makefile orchestration for chosen-run deployment sequence</title>
  <action>
    Add documented Makefile targets and a short deployment runbook so the user can provide a run/checkpoint and execute the ordered sequence: convert checkpoint to safetensors, export ONNX, simplify/check size, fit/apply validation-only temperature scaling, write or update the active Gradio artifact config/pointer, then run an inference/UI smoke check. Targets should be explicit and composable, for example `make checkpoint-to-safetensors RUN=...`, `make export-model RUN=...`, `make calibrate-model RUN=...`, `make prepare-demo-artifact RUN=...`, and a higher-level `make deploy-run-to-demo RUN=...` that chains the safe steps. Exact names may differ, but the order must be discoverable from `make help`.
  </action>
  <acceptance_criteria>
    - `make help` lists the deployment/demo artifact targets and required variables such as RUN, CHECKPOINT, SAFETENSORS, or ARTIFACT_CONFIG.
    - A single high-level Make target exists or is documented to perform the safe ordered sequence for a chosen run/checkpoint.
    - The Makefile targets call scripts rather than embedding complex logic directly in Make.
    - The final model swap is a config/artifact pointer update; Gradio prediction code does not need manual editing.
  </acceptance_criteria>
</task>

</tasks>

<verification>
Use `04-VALIDATION.md`. For code changes, run targeted tests plus the smallest safetensors round-trip, ONNX, and calibration smoke commands available. Also verify `make help` exposes the ordered deployment/demo artifact targets.
</verification>
