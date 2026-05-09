---
phase: 6
phase_name: Hugging Face Spaces Demo Deploy
branch: feat/hugging-face-spaces-deploy
plan_id: 06-02
objective: Document and optionally automate Hugging Face Spaces deployment
wave: 2
depends_on: ['06-01']
requirements_addressed: ['DEPLOY-04', 'DEPLOY-05']
files_modified: ['Makefile', 'docs/hf_spaces_deploy.md', 'deploy/hf_space/README.md']
autonomous: true
---

# Plan 02 — Document and optionally automate HF Spaces deployment

<objective>
Provide step-by-step manual deployment instructions and optional CLI deployment commands for publishing the packaged Gradio ONNX demo to Hugging Face Spaces.
</objective>

<must_haves>
- Covers requirement IDs: DEPLOY-04, DEPLOY-05.
- Default path is Hugging Face Spaces Gradio SDK on CPU Basic, not Docker.
- CLI deploy is optional because it may require Hugging Face login/token.
</must_haves>

<tasks>

<task id="P6-02-T1" type="execute">
  <title>Write manual deploy runbook</title>
  <action>
    Document manual steps: create a Hugging Face Space, choose Gradio SDK, CPU Basic hardware, push/upload deploy package, inspect build logs, open/test public URL, and record link for Phase 7 report.
  </action>
  <acceptance_criteria>
    - Runbook is understandable without prior Hugging Face Spaces experience.
    - Runbook says Docker is not required for this project.
  </acceptance_criteria>
</task>

<task id="P6-02-T2" type="execute">
  <title>Add optional CLI deploy path</title>
  <action>
    If safe and supported by local tools, document or add a Makefile target for CLI deploy using Hugging Face auth. If auth is missing, provide exact manual fallback rather than blocking.
  </action>
  <acceptance_criteria>
    - CLI path names required variables such as `SPACE=owner/name`.
    - Missing auth/token behavior is documented.
  </acceptance_criteria>
</task>

</tasks>

<verification>
Use `06-VALIDATION.md`; verify docs mention Gradio SDK, CPU Basic, ONNX Runtime, and no Docker.
</verification>
