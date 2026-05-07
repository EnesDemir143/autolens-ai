---
phase: 2
phase_name: Dataset Research and Curation
branch: feat/dataset-curation-eda
plan_id: 02-03
objective: Map raw labels and curate the 8-class dataset
wave: 2
depends_on: ['02-01', '02-02']
requirements_addressed: ['DATA-03', 'DATA-05']
files_modified: ['src/autolens_ai/data/', 'configs/data/', 'artifacts/dataset/', 'docs/dataset_manifest.md']
autonomous: true
---

# Plan 03 — Map raw labels and curate the 8-class dataset

<objective>
Map raw labels and curate the 8-class dataset
</objective>

<must_haves>
- Covers requirement IDs: DATA-03, DATA-05. Curation target is approximately 32k clean images, about 4k/class, from 36k–40k raw candidates.
- Covers phase decisions: D-01, D-02, D-03, D-04, D-05 where relevant to this plan.
- Produces concrete evidence for downstream phases.
</must_haves>

<threat_model>
Keep local credentials and dataset/API tokens out of git. Use `.env.example` placeholders only. Do not upload private data or final instructor test data into training artifacts. Validate all generated files are local project artifacts before committing.
</threat_model>

<tasks>

<task id="P2-03-T1" type="execute">
  <title>Read phase context and prepare target files</title>
  <read_first>
    - `.planning/phases/02-dataset-research-and-curation/02-CONTEXT.md`
    - `.planning/phases/02-dataset-research-and-curation/02-RESEARCH.md`
    - `.planning/REQUIREMENTS.md`
    - `.planning/ROADMAP.md`
  </read_first>
  <action>
    Create or update the minimal files needed for this plan objective: `Map raw labels and curate the 8-class dataset`. Keep names explicit and avoid hidden behavior. Preserve outputs from earlier phases.
  </action>
  <acceptance_criteria>
    - `test -d .planning/phases/02-dataset-research-and-curation` exits 0.
    - At least one project file or documentation artifact relevant to `03-label-curation` exists after execution.
  </acceptance_criteria>
</task>

<task id="P2-03-T2" type="execute">
  <title>Implement or document the planned capability</title>
  <read_first>
    - `.planning/phases/02-dataset-research-and-curation/02-VALIDATION.md`
    - `AGENTS.md`
  </read_first>
  <action>
    Implement the smallest working slice for `Map raw labels and curate the 8-class dataset`. The label mapping must avoid unsafe direct mappings: `City Car` must be filtered before `MICRO`/`HATCHBACK`, generic `Truck` must be filtered before `PICK_UP`, and `Crossover`/`MPV`/`Coupe` must not be blindly mapped. If the slice depends on external credentials, network access, dataset availability, gated model access, or prior phase runtime artifacts, document the exact blocker and fallback in the corresponding docs/artifacts file instead of guessing.
  </action>
  <acceptance_criteria>
    - Output references all requirement IDs: DATA-03, DATA-05.
    - Output contains no instructor final test data.
    - Any external dependency blocker is documented with the exact command or resource ID.
  </acceptance_criteria>
</task>

<task id="P2-03-T3" type="execute">
  <title>Verify and record evidence</title>
  <read_first>
    - `.planning/phases/02-dataset-research-and-curation/02-VALIDATION.md`
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
Use the checks listed in `02-VALIDATION.md` for this phase and run project quality gates when implementation files are changed.
</verification>
