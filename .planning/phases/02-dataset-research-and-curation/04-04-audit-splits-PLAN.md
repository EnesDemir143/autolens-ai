---
phase: 2
phase_name: Dataset Research and Curation
branch: feat/dataset-research-and-curation
plan_id: 02-04
objective: Generate manifest, EDA/outlier audit, balance audit, dedup report, and split files
wave: 2
depends_on: ['02-01', '02-02']
requirements_addressed: ['DATA-04', 'DATA-05', 'DATA-06']
files_modified: ['src/autolens_ai/data/', 'configs/data/', 'artifacts/dataset/', 'docs/dataset_manifest.md']
autonomous: true
---

# Plan 04 — Generate manifest, EDA/outlier audit, balance audit, dedup report, and split files

<objective>
Generate manifest, EDA/outlier audit, balance audit, dedup report, and split files
</objective>

<must_haves>
- Covers requirement IDs: DATA-04, DATA-05, DATA-06. Audit must report target-vs-actual counts for the ~4k/class, ~32k clean final dataset goal.
- Audit must include per-dataset EDA, merged EDA, class distribution analysis, source-by-class coverage, and explicit outlier/anomaly review before final split files are accepted.
- Covers phase decisions: D-01, D-02, D-03, D-04, D-05 where relevant to this plan.
- Produces concrete evidence for downstream phases.
</must_haves>

<threat_model>
Keep local credentials and dataset/API tokens out of git. Use `.env.example` placeholders only. Do not upload private data or final instructor test data into training artifacts. Validate all generated files are local project artifacts before committing.
</threat_model>

<tasks>

<task id="P2-04-T1" type="execute">
  <title>Read phase context and prepare target files</title>
  <read_first>
    - `.planning/phases/02-dataset-research-and-curation/02-CONTEXT.md`
    - `.planning/phases/02-dataset-research-and-curation/02-RESEARCH.md`
    - `.planning/REQUIREMENTS.md`
    - `.planning/ROADMAP.md`
  </read_first>
  <action>
    Create or update the minimal files needed for this plan objective: `Generate manifest, EDA/outlier audit, balance audit, dedup report, and split files`. Keep names explicit and avoid hidden behavior. Preserve outputs from earlier phases.
  </action>
  <acceptance_criteria>
    - `test -d .planning/phases/02-dataset-research-and-curation` exits 0.
    - At least one project file or documentation artifact relevant to `04-audit-splits` exists after execution.
  </acceptance_criteria>
</task>

<task id="P2-04-T2" type="execute">
  <title>Implement or document the planned capability</title>
  <read_first>
    - `.planning/phases/02-dataset-research-and-curation/02-VALIDATION.md`
    - `AGENTS.md`
  </read_first>
  <action>
    Implement the smallest working slice for `Generate manifest, EDA/outlier audit, balance audit, dedup report, and split files`. Produce EDA artifacts under `artifacts/dataset/eda/` covering: per-source class counts, merged class distribution, source-by-class table, target-vs-actual counts, missing/null label and metadata counts, missing referenced image-file counts, image dimensions/aspect ratios/file-size summaries, invalid/corrupt image counts, duplicate/hash results, ambiguous-label counts, and outlier/anomaly review candidate lists. Add a dataset decision note that explains which datasets/classes/images are accepted, filtered, excluded, or deferred and what preprocessing risks should be revisited in Phase 3. If the slice depends on external credentials, network access, dataset availability, gated model access, or prior phase runtime artifacts, document the exact blocker and fallback in the corresponding docs/artifacts file instead of guessing.
  </action>
  <acceptance_criteria>
    - Output references all requirement IDs: DATA-04, DATA-05, DATA-06.
    - Per-source and merged EDA artifacts exist or exact blockers are documented.
    - Outlier/anomaly review artifacts exist and are linked from the dataset decision note.
    - Split generation is gated on the EDA decision note, not only on raw counts.
    - Output contains no instructor final test data.
    - Any external dependency blocker is documented with the exact command or resource ID.
  </acceptance_criteria>
</task>

<task id="P2-04-T3" type="execute">
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
