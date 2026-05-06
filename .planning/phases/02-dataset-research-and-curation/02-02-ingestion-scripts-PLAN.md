---
phase: 2
phase_name: Dataset Research and Curation
plan_id: 02-02
objective: Create source download/ingestion command surfaces
wave: 1
depends_on: []
requirements_addressed: ['DATA-01', 'DATA-02']
files_modified: ['src/autolens_ai/data/', 'configs/data/', 'artifacts/dataset/', 'docs/dataset_manifest.md']
autonomous: true
---

# Plan 02 — Create source download/ingestion command surfaces

<objective>
Create source download/ingestion command surfaces
</objective>

<must_haves>
- Covers requirement IDs: DATA-01, DATA-02.
- Covers phase decisions: D-01, D-02, D-03, D-04, D-05 where relevant to this plan.
- Produces concrete evidence for downstream phases.
</must_haves>

<threat_model>
Keep local credentials and dataset/API tokens out of git. Use `.env.example` placeholders only. Do not upload private data or final instructor test data into training artifacts. Validate all generated files are local project artifacts before committing.
</threat_model>

<tasks>

<task id="P2-02-T1" type="execute">
  <title>Read phase context and prepare target files</title>
  <read_first>
    - `.planning/phases/02-dataset-research-and-curation/02-CONTEXT.md`
    - `.planning/phases/02-dataset-research-and-curation/02-RESEARCH.md`
    - `.planning/REQUIREMENTS.md`
    - `.planning/ROADMAP.md`
  </read_first>
  <action>
    Create or update the minimal files needed for this plan objective: `Create source download/ingestion command surfaces`. Keep names explicit and avoid hidden behavior. Preserve outputs from earlier phases.
  </action>
  <acceptance_criteria>
    - `test -d .planning/phases/02-dataset-research-and-curation` exits 0.
    - At least one project file or documentation artifact relevant to `02-ingestion-scripts` exists after execution.
  </acceptance_criteria>
</task>

<task id="P2-02-T2" type="execute">
  <title>Implement or document the planned capability</title>
  <read_first>
    - `.planning/phases/02-dataset-research-and-curation/02-VALIDATION.md`
    - `AGENTS.md`
  </read_first>
  <action>
    Implement the smallest working slice for `Create source download/ingestion command surfaces`. If the slice depends on external credentials, network access, dataset availability, gated model access, or prior phase runtime artifacts, document the exact blocker and fallback in the corresponding docs/artifacts file instead of guessing.
  </action>
  <acceptance_criteria>
    - Output references all requirement IDs: DATA-01, DATA-02.
    - Output contains no instructor final test data.
    - Any external dependency blocker is documented with the exact command or resource ID.
  </acceptance_criteria>
</task>

<task id="P2-02-T3" type="execute">
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
