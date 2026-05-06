# AutoLens AI Agent Rules

These rules apply to all agent work in this repository. They exist so future agents use the planning graph, GSD phase plans, and local skills consistently instead of re-deriving decisions from scratch.

## 1. Start From GSD Planning State

Before implementation, review the current GSD state:

1. `.planning/PROJECT.md` — project goal, constraints, and key decisions.
2. `.planning/ROADMAP.md` — six-phase execution order.
3. `.planning/REQUIREMENTS.md` — source-of-truth requirement IDs.
4. `.planning/STATE.md` — current phase readiness/status.
5. The active phase directory under `.planning/phases/NN-*`:
   - `NN-CONTEXT.md`
   - `NN-RESEARCH.md`
   - `NN-VALIDATION.md`
   - `*-PLAN.md`
   - `NN-UI-SPEC.md` when present, especially Phase 5.

Do not bypass GSD execution for planned work. Use the relevant phase plan and keep requirement coverage intact.

## 2. Use Graphify Before Broad Project Reasoning

The project knowledge graph is already built from GSD planning artifacts.

Read or query these before broad reasoning, architecture changes, phase handoffs, or dependency-impact questions:

- `.planning/graphs/GRAPH_REPORT.md`
- `.planning/graphs/graph.json`
- `.planning/graphs/graph.html` when visual inspection helps

Before every Graphify read/query/use, refresh the graph from the current workspace state:

```fish
node "$HOME/.codex/get-shit-done/bin/gsd-tools.cjs" graphify build .
```

Preferred query command after the update:

```fish
node "$HOME/.codex/get-shit-done/bin/gsd-tools.cjs" graphify query "<term>"
```

Shortcut through the project Makefile:

```fish
make graphify-update
```

Useful terms:

- `DINOv3`
- `Dataset`
- `Gradio`
- `MobileNetV4`
- `F1-first evaluation`
- `IEEE LaTeX report`
- `95 MB model limit`

Do not rely on stale graph output: run `graphify build .` first, then read/query graph artifacts so source-code and planning relationships are current.

## 3. Always Apply Karpathy Guidelines

Before writing or changing code, read and follow:

- `.agents/skills/andrej-karpathy-skills/skills/karpathy-guidelines/SKILL.md`

Operational interpretation for this project:

- Keep code small, explicit, and explainable line by line for demo questions.
- Do not add speculative abstractions.
- Do not implement future-phase behavior early.
- Prefer tests and verifiable outputs over assumptions.
- Every changed line should trace to a requirement ID or phase plan task.

## 4. Use Local Skills When Relevant

Local skills are part of the project contract. Use the needed skill instead of reinventing its workflow.

Available project skills:

- Karpathy coding discipline:
  - `.agents/skills/andrej-karpathy-skills/skills/karpathy-guidelines/SKILL.md`
- IEEE report generation:
  - `.agents/skills/latex-report-creator/SKILL.md`
- Architecture diagrams when requested:
  - `.agents/skills/architecture-diagram-generator/architecture-diagram/SKILL.md`

Report rule:

- Do not generate the final IEEE report before Phase 6.
- In Phase 6, use the LaTeX report creator skill and actual implementation evidence.
- The report must avoid raw code dumps and use academic prose/pseudocode.

Diagram rule:

- If the user asks for an architecture/system diagram, use the architecture diagram skill and base the diagram on `.planning/ROADMAP.md`, `.planning/graphs/graph.json`, and implemented evidence if available.

## 5. Preserve Phase Boundaries

The roadmap is sequential and fixed unless the user explicitly changes it:

1. Project Foundation
2. Dataset Research and Curation
3. Baseline Training Pipeline
4. Model Comparison and Selection
5. Gradio Demo Interface
6. Final Evidence and IEEE Report

Do not implement later-phase capabilities early:

- No dataset curation in Phase 1.
- No model training before Phase 2 dataset artifacts exist.
- No DINOv3/LoRA comparison before baseline training exists.
- No final report before metrics, plots, UI evidence, and final model selection exist.

## 6. Respect Core Technical Decisions

Hard project decisions:

- Python 3.12 with uv.
- Sequential GSD execution.
- F1-score is the primary model-selection metric.
- Final deployable model artifact must stay under 95 MB.
- Target classes are exactly:
  - SUV
  - VAN
  - STATION WAGON
  - MICRO
  - OPEN WHEEL / F1
  - SEDAN
  - HATCHBACK
  - PICK UP
- Model candidates:
  - MobileNetV4 Conv Medium
  - EfficientNet-B2
  - ResNet baseline
  - DINOv3 ViT-S/16 (`facebook/dinov3-vits16-pretrain-lvd1689m`) with non-LoRA first and LoRA optional.
- UI is Gradio Blocks with a clean modern presentation-ready layout.

## 7. Dependency Rules

Use `docs/plan.md`, `.planning/research/STACK.md`, and future `docs/dependencies.md` as dependency sources of truth.

Expected install style:

```fish
uv add ...
uv add --dev ...
```

Do not add new dependencies unless:

1. A phase plan requires it, or
2. The user explicitly requests it, or
3. A blocker is documented with rationale and the dependency is the smallest safe fix.

If adding a dependency, document why and where it is used.


## 8. Experiment Tracking and Artifact Logging Rules

Weights & Biases (W&B) is the primary experiment tracking system for AutoLens AI. Treat it as the project MLflow-equivalent experiment ledger.

Decision: Use W&B as the primary tracker. Do not add MLflow, MLflow tracking abstractions, or duplicate experiment-tracking code unless W&B becomes blocked by access/quota/sync/export/policy issues or the user explicitly requests an MLflow migration.

Use W&B for all training, evaluation, model-comparison, and final-selection runs starting from the first executable training phase.

Required W&B setup:

- Do not hardcode API keys or tokens in source, configs, notebooks, or docs.
- Authenticate locally with `uv run wandb login` or provide `WANDB_API_KEY` through the environment.
- Keep `.env` untracked; use `.env.example` only for placeholder variable names.
- Prefer a stable project name such as `autolens-ai` unless the user explicitly changes it.

Every training/evaluation run must log:

- Run identity: phase, plan ID when relevant, model name, dataset version/manifest path, split version, git commit or dirty-worktree note, seed, device (`mps` or `cpu`), and config file path.
- Hyperparameters: learning rate, optimizer, scheduler, batch size, image size, epochs, augmentations, early-stopping settings, class weights/sampling settings if used.
- Required metrics: Accuracy, Precision, Recall, macro F1, weighted F1, per-class metrics, validation loss, validation accuracy, and final selected ranking metric.
- Required plots/artifacts: training/validation loss curve, training/validation accuracy curve, normalized 8x8 confusion matrix, class distribution/balance report, and model-size/latency comparison when available.
- Dataset evidence: source manifest, class mapping, split files/checksums, dedup/balance audit outputs, and notes for weak classes such as MICRO, STATION WAGON, and OPEN WHEEL/F1.
- Model artifacts: checkpoints or exported artifacts needed to reproduce the run, final class mapping, preprocessing config, and artifact-size measurement. Do not upload private instructor final test data.
- UI/final demo evidence in later phases: selected model artifact, inference preprocessing metadata, representative non-private sample predictions, and probability outputs used for the final report.

Rules for W&B artifacts:

- Store report-relevant generated files under local `artifacts/`, `outputs/`, or phase-specific evidence directories first, then log them to W&B.
- Name artifacts with model, dataset/split version, phase, and date where practical.
- Mark the final candidate model explicitly in W&B and verify it remains under the 95 MB submission limit before treating it as deployable.
- Never log credentials, `.env`, Kaggle tokens, Hugging Face tokens, instructor-provided final test images, or private raw data.

If W&B is unavailable:

- Do not block local training solely because W&B is offline.
- Use W&B offline mode or write local evidence files, then document the exact blocker in the phase summary/report.
- Sync/log later when credentials or network are available.

## 9. Dataset Safety Rules

- Never use instructor-provided final test images during training or tuning.
- Record every dataset source, URL/ID, license note, class mapping, and image count.
- Treat MICRO, STATION WAGON, and OPEN WHEEL/F1 as likely gap classes until audited.
- Deduplicate and split before training.
- Keep source manifests report-ready.

## 10. Verification Before Completion

For any phase execution, completion requires evidence from that phase's `NN-VALIDATION.md`.

Minimum expectations:

- Run the smallest relevant tests/checks.
- Capture exact failures if network, Kaggle, Hugging Face gated access, or package wheels block progress.
- Update phase evidence/docs so Phase 6 can build the final report.
- Do not claim a phase is complete only because files were created; verify behavior or document blockers.

## 11. Phase Report Hygiene

After every GSD phase is completed or handed off, create/update that phase's report under `docs/phase_reports/` using:

- `docs/phase_reports/_TEMPLATE_PHASE_REPORT.md`

Rules:

- Use the template structure as the required report format.
- Name reports consistently by phase, e.g. `docs/phase_reports/phase_01_project_foundation.md`.
- Include the phase status, short outcome summary, commits, changed/added files, validation commands and results, extra work outside the original plan, notes, and next step.
- Do not mark the phase report as complete without evidence from the phase `NN-VALIDATION.md` checks or clearly documented blockers.
- Keep the phase report factual and evidence-based so Phase 6 can reuse it for the final IEEE report.

## 12. Git and Documentation Hygiene

- Keep planning docs and phase artifacts committed when changed.
- Use the repository Lore commit protocol from AGENTS.md for commit messages.
- Do not commit `.DS_Store`, credentials, raw private datasets, or instructor final test data.
- Keep README and docs aligned with actual commands that work in fish.

## 13. When Unsure

Use this order:

1. Query/read `.planning/graphs/`.
2. Read current phase `CONTEXT`, `RESEARCH`, `VALIDATION`, and `PLAN` files.
3. Read the relevant local skill from `.agents/skills/`.
4. Inspect actual code/artifacts if implementation exists.
5. Ask the user only if the decision would change scope, require credentials, or risk destructive work.
