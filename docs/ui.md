# AutoLens AI Gradio Demo

## Quick Start

```bash
# Launch the demo locally
make demo

# Run a non-interactive smoke test
make demo-smoke
```

The demo will be available at `http://localhost:7860`.

## Architecture

- **Model backend**: ONNX Runtime loads the artifact specified in `artifacts/demo/active_model.json`.
- **UI framework**: Gradio Blocks (single-page, drag-and-drop upload).
- **Predictor**: `autolens_ai.inference.ONNXPredictor` handles preprocessing, inference, and temperature-scaled softmax.

## Swapping the Model

To point the demo to a different exported/calibrated artifact:

1. Run the Phase 4 deployment sequence for the new checkpoint:
   ```bash
   make deploy-run-to-demo CHECKPOINT=checkpoints/your_run/best.ckpt
   ```
2. Or manually update `artifacts/demo/active_model.json`:
   ```bash
   make prepare-demo-artifact \
       METADATA=artifacts/export/other_run/metadata.json \
       ONNX=artifacts/export/other_run/model.onnx \
       CALIBRATION=artifacts/export/other_run/calibration.json
   ```
3. Restart the demo (no code changes needed).

## UI Features

- Drag-and-drop or click to upload a vehicle image.
- Preview appears immediately after upload.
- Click **Classify** (or wait for auto-classify on upload).
- During inference, a loading message appears so the UI does not look frozen.
- Results show:
  - Predicted class name
  - Confidence score
  - Inference latency
  - Horizontal bar chart of all 8 class probabilities

## Requirements Covered

| Requirement | Status | Evidence |
|---|---|---|
| UI-01 Upload/drag-and-drop | Complete | `app.py` `gr.Image` with `sources=["upload", "clipboard"]` |
| UI-02 Image preview | Complete | `gr.Image` renders the uploaded image |
| UI-03 Classification trigger | Complete | `gr.Button("Classify")` + auto-trigger on upload |
| UI-04 Predicted class display | Complete | `gr.Textbox` shows predicted class |
| UI-05 Confidence + probability chart | Complete | `gr.Textbox` + `gr.Plot` bar chart |
| UI-06 Clean modern layout | Complete | Gradio Blocks two-column layout |
| UI-07 Acceptable latency | Complete | Smoke test ~15 ms avg on M2 Pro |
| UI-08 Loading/progress state | Complete | Markdown status visible during inference |

## Smoke Test Evidence

Latest run: `artifacts/demo/smoke_test_evidence.json`

## Troubleshooting

| Issue | Fix |
|---|---|
| `FileNotFoundError: active_model.json` | Run `make prepare-demo-artifact` or `make deploy-run-to-demo` |
| ONNX model missing | Run Phase 4 export sequence (`make export-model`) |
| Port 7860 in use | Edit `app.py` `server_port` or kill existing process |
