---
phase: 3
phase_name: Baseline Training Pipeline
branch: feat/baseline-training-pipeline
plan_id: 03-04
objective: Add baseline experiment configs and runbook for the CNN baseline models
wave: 2
depends_on: ['03-01', '03-02']
requirements_addressed: ['TRN-01', 'TRN-02', 'TRN-03']
files_modified: ['src/autolens_ai/training/', 'src/autolens_ai/models/', 'configs/experiments/', 'docs/training.md']
autonomous: true
---

# Plan 04 — Add baseline experiment configs and runbook for the CNN baseline models

<objective>
Add baseline experiment configs and runbook for the CNN baseline models
</objective>

<must_haves>
- Covers requirement IDs: TRN-01, TRN-02, TRN-03.
- Covers phase decisions: D-01, D-02, D-03, D-04, D-05, D-06, D-07 where relevant to this plan.
- Produces concrete evidence for downstream phases.
</must_haves>

<threat_model>
Keep local credentials and dataset/API tokens out of git. Use `.env.example` placeholders only. Do not upload private data or final instructor test data into training artifacts. Validate all generated files are local project artifacts before committing.
</threat_model>

<tasks>

<task id="P3-04-T1" type="execute">
  <title>Read phase context and prepare target files</title>
  <read_first>
    - `.planning/phases/03-baseline-training-pipeline/03-CONTEXT.md`
    - `.planning/phases/03-baseline-training-pipeline/03-RESEARCH.md`
    - `.planning/REQUIREMENTS.md`
    - `.planning/ROADMAP.md`
  </read_first>
  <action>
    Create or update the minimal files needed for this plan objective: `Add baseline experiment configs and runbook for the CNN baseline models`. Keep names explicit and avoid hidden behavior. Preserve outputs from earlier phases.
  </action>
  <acceptance_criteria>
    - `test -d .planning/phases/03-baseline-training-pipeline` exits 0.
    - At least one project file or documentation artifact relevant to `04-baseline-runbook` exists after execution.
  </acceptance_criteria>
</task>

<task id="P3-04-T2" type="execute">
  <title>Implement or document the planned capability</title>
  <read_first>
    - `.planning/phases/03-baseline-training-pipeline/03-VALIDATION.md`
    - `AGENTS.md`
  </read_first>
  <action>
    Implement the smallest working slice for `Add baseline experiment configs and runbook for the CNN baseline models`. The runbook must define the experiment order for MobileNetV4 Conv Medium, EfficientNet-B2, and ResNet18: Baseline 0 no augmentation with class-weighted CrossEntropy on the Phase 2 outlier-unfiltered split; Baseline 1 light augmentation comparison; Baseline 2 class-weighted loss versus weighted sampler; optional low-priority Baseline 3 outlier/near-duplicate filtered split comparison only if earlier results or leakage review justify the extra data loss. It must also state that Phase 4 adds DINOv3 non-LoRA and optional LoRA as the fourth model family for final comparison. If the slice depends on external credentials, network access, dataset availability, gated model access, or prior phase runtime artifacts, document the exact blocker and fallback in the corresponding docs/artifacts file instead of guessing.
  </action>
  <acceptance_criteria>
    - Output references all requirement IDs: TRN-01, TRN-02, TRN-03.
    - Output contains no instructor final test data.
    - The runbook separates no-augmentation, augmentation, imbalance, and optional outlier-filtered variants instead of mixing them in one baseline.
    - The runbook names macro F1, weighted F1, per-class F1, and normalized confusion matrix as comparison evidence.
    - Any external dependency blocker is documented with the exact command or resource ID.
  </acceptance_criteria>
</task>

<task id="P3-04-T3" type="execute">
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
