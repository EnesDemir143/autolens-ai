---
phase: 3
phase_name: Baseline Training Pipeline
branch: feat/baseline-training-pipeline
plan_id: 03-01
objective: Implement dataset class/datamodule and preprocessing config
wave: 1
depends_on: []
requirements_addressed: ['TRN-06', 'TRN-07']
files_modified: ['src/autolens_ai/training/', 'src/autolens_ai/models/', 'configs/experiments/', 'docs/training.md']
autonomous: true
---

# Plan 01 — Implement dataset class/datamodule and preprocessing config

<objective>
Implement dataset class/datamodule and preprocessing config
</objective>

<must_haves>
- Covers requirement IDs: TRN-06, TRN-07.
- Covers phase decisions: D-01, D-02, D-03, D-04, D-05, D-06, D-07 where relevant to this plan.
- Produces concrete evidence for downstream phases.
</must_haves>

<threat_model>
Keep local credentials and dataset/API tokens out of git. Use `.env.example` placeholders only. Do not upload private data or final instructor test data into training artifacts. Validate all generated files are local project artifacts before committing.
</threat_model>

<tasks>

<task id="P3-01-T1" type="execute">
  <title>Read phase context and prepare target files</title>
  <read_first>
    - `.planning/phases/03-baseline-training-pipeline/03-CONTEXT.md`
    - `.planning/phases/03-baseline-training-pipeline/03-RESEARCH.md`
    - `.planning/REQUIREMENTS.md`
    - `.planning/ROADMAP.md`
  </read_first>
  <action>
    Create or update the minimal files needed for this plan objective: `Implement dataset class/datamodule and preprocessing config`. Keep names explicit and avoid hidden behavior. Preserve outputs from earlier phases.
  </action>
  <acceptance_criteria>
    - `test -d .planning/phases/03-baseline-training-pipeline` exits 0.
    - At least one project file or documentation artifact relevant to `01-data-module` exists after execution.
  </acceptance_criteria>
</task>

<task id="P3-01-T2" type="execute">
  <title>Implement or document the planned capability</title>
  <read_first>
    - `.planning/phases/03-baseline-training-pipeline/03-VALIDATION.md`
    - `AGENTS.md`
  </read_first>
  <action>
    Implement the smallest working slice for `Implement dataset class/datamodule and preprocessing config`. The first preprocessing config must define Baseline 0 with no train augmentation: resize 256, center crop 224, tensor conversion, and ImageNet normalization for train/validation/internal-test. It must consume Phase 2 split CSVs rather than raw folders directly. If the slice depends on external credentials, network access, dataset availability, gated model access, or prior phase runtime artifacts, document the exact blocker and fallback in the corresponding docs/artifacts file instead of guessing.
  </action>
  <acceptance_criteria>
    - Output references all requirement IDs: TRN-06, TRN-07.
    - Output contains no instructor final test data.
    - A no-augmentation Baseline 0 preprocessing config exists and is clearly separate from later augmentation variants.
    - The datamodule can be pointed at `artifacts/dataset/splits/all_splits.csv` or the split-specific CSV files.
    - Any external dependency blocker is documented with the exact command or resource ID.
  </acceptance_criteria>
</task>

<task id="P3-01-T3" type="execute">
  <title>Verify and record evidence</title>
  <read_first>
    - `.planning/phases/03-baseline-training-pipeline/03-VALIDATION.md`
  </read_first>
  <action>
    Run the smallest verification command(s) available for this plan. Save or document evidence in the phase artifact area so later phases and the final IEEE report can consume it.
  </action>
  <acceptance_criteria>
    - Verification evidence exists in docs, artifacts, test output, or a phase summary.
    - Evidence is specific enough to support the final report.
  </acceptance_criteria>
</task>

</tasks>

<verification>
Use the checks listed in `03-VALIDATION.md` for this phase and run project quality gates when implementation files are changed.
</verification>
