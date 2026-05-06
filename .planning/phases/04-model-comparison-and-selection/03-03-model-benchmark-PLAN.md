---
phase: 4
phase_name: Model Comparison and Selection
plan_id: 04-03
objective: Benchmark all candidate models by macro F1, size, and latency
wave: 2
depends_on: ['04-01', '04-02']
requirements_addressed: ['EVAL-06', 'EVAL-07']
files_modified: ['src/autolens_ai/evaluation/', 'artifacts/evaluation/', 'docs/model_selection.md']
autonomous: true
---

# Plan 03 — Benchmark all candidate models by macro F1, size, and latency

<objective>
Benchmark all candidate models by macro F1, size, and latency
</objective>

<must_haves>
- Covers requirement IDs: EVAL-06, EVAL-07.
- Covers phase decisions: D-01, D-02, D-03, D-04, D-05 where relevant to this plan.
- Produces concrete evidence for downstream phases.
</must_haves>

<threat_model>
Keep local credentials and dataset/API tokens out of git. Use `.env.example` placeholders only. Do not upload private data or final instructor test data into training artifacts. Validate all generated files are local project artifacts before committing.
</threat_model>

<tasks>

<task id="P4-03-T1" type="execute">
  <title>Read phase context and prepare target files</title>
  <read_first>
    - `.planning/phases/04-model-comparison-and-selection/04-CONTEXT.md`
    - `.planning/phases/04-model-comparison-and-selection/04-RESEARCH.md`
    - `.planning/REQUIREMENTS.md`
    - `.planning/ROADMAP.md`
  </read_first>
  <action>
    Create or update the minimal files needed for this plan objective: `Benchmark all candidate models by macro F1, size, and latency`. Keep names explicit and avoid hidden behavior. Preserve outputs from earlier phases.
  </action>
  <acceptance_criteria>
    - `test -d .planning/phases/04-model-comparison-and-selection` exits 0.
    - At least one project file or documentation artifact relevant to `03-model-benchmark` exists after execution.
  </acceptance_criteria>
</task>

<task id="P4-03-T2" type="execute">
  <title>Implement or document the planned capability</title>
  <read_first>
    - `.planning/phases/04-model-comparison-and-selection/04-VALIDATION.md`
    - `AGENTS.md`
  </read_first>
  <action>
    Implement the smallest working slice for `Benchmark all candidate models by macro F1, size, and latency`. If the slice depends on external credentials, network access, dataset availability, gated model access, or prior phase runtime artifacts, document the exact blocker and fallback in the corresponding docs/artifacts file instead of guessing.
  </action>
  <acceptance_criteria>
    - Output references all requirement IDs: EVAL-06, EVAL-07.
    - Output contains no instructor final test data.
    - Any external dependency blocker is documented with the exact command or resource ID.
  </acceptance_criteria>
</task>

<task id="P4-03-T3" type="execute">
  <title>Verify and record evidence</title>
  <read_first>
    - `.planning/phases/04-model-comparison-and-selection/04-VALIDATION.md`
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
Use the checks listed in `04-VALIDATION.md` for this phase and run project quality gates when implementation files are changed.
</verification>
