#!/usr/bin/env python
from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Hugging Face Spaces runs app.py directly from /app and does not install the
# local src/ package automatically. Make the copied source tree importable at
# startup before importing autolens_ai.
APP_ROOT = Path(__file__).resolve().parent
SRC_ROOT = APP_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autolens_ai.inference import ONNXPredictor
from scripts.download_from_hf import download_model, write_active_model

PREDICTOR: ONNXPredictor | None = None


def _ensure_model_downloaded() -> None:
    active_model_path = Path("artifacts/demo/active_model.json")
    if active_model_path.exists():
        print("Model artifacts already present.", flush=True)
        return

    print("Downloading model artifacts...", flush=True)
    export_dir = download_model("dinov3-weighted", force=False)
    write_active_model(export_dir)
    print("Model artifacts ready.", flush=True)


def _load_predictor() -> ONNXPredictor:
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = ONNXPredictor.from_default_config()
    return PREDICTOR


def _build_prob_chart(probabilities: dict[str, float]) -> plt.Figure:
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    probs = [item[1] for item in sorted_items]
    colors = ["#6b85d9", "#3a4b7f", "#3d4f60"] + ["#252930"] * max(0, len(labels) - 3)

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    fig.patch.set_facecolor("#12151b")
    ax.set_facecolor("#12151b")

    bars = ax.barh(range(len(labels)), probs, color=colors[: len(labels)], height=0.52, edgecolor="none")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color="#929bb0", family="monospace")
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors="#535c72", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#29303c", alpha=0.35, linewidth=0.7)

    for idx, (bar, prob) in enumerate(zip(bars, probs, strict=True)):
        ax.text(
            bar.get_width() + 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center",
            fontsize=8.5,
            color="#e1e4ec" if idx < 3 else "#929bb0",
            fontweight="bold" if idx == 0 else "normal",
            family="monospace",
        )
        if idx < 3:
            ax.text(
                0.008,
                bar.get_y() + bar.get_height() / 2,
                f"#{idx + 1}",
                va="center",
                fontsize=7.5,
                color="#0b0d10",
                fontweight="bold",
                family="monospace",
            )

    plt.tight_layout(pad=0.45)
    return fig


def _result_card(result: dict[str, Any]) -> str:
    pred = result["predicted_display"]
    conf = float(result["confidence"])
    latency = float(result["latency_ms"])
    conf_color = "#38b47f" if conf >= 0.85 else ("#e0ac42" if conf >= 0.60 else "#e25859")
    conf_tag = "HIGH" if conf >= 0.85 else ("MODERATE" if conf >= 0.60 else "LOW")
    return f"""
    <div class="result-card fade-up">
      <div class="eyebrow">PREDICTED CLASS</div>
      <div class="prediction-title">{pred}</div>
      <div class="metric-row">
        <div class="metric-pill" style="border-color:{conf_color}55;background:{conf_color}14">
          <span style="color:{conf_color}">{conf:.1%}</span>
          <small>CONFIDENCE · {conf_tag}</small>
        </div>
        <div class="metric-pill">
          <span>{latency:.1f} ms</span>
          <small>ONNX RUNTIME</small>
        </div>
      </div>
    </div>
    """


def classify(image: Image.Image | None) -> tuple[str, Any]:
    if image is None:
        return (
            "<div class='empty-card'>Select a vehicle image first.</div>",
            gr.update(visible=False),
        )
    try:
        predictor = _load_predictor()
        result = predictor.predict(image)
        chart = _build_prob_chart(result["probabilities"])
        return (
            _result_card(result),
            gr.update(value=chart, visible=True),
        )
    except Exception as exc:
        return (
            f"<div class='error-card'>Error: {exc}</div>",
            gr.update(visible=False),
        )


def reset_outputs() -> tuple[str, Any]:
    return (
        "<div class='empty-card'>Upload an image, then press Classify.</div>",
        gr.update(value=None, visible=False),
    )


CSS = """
:root {
  --autolens-bg: #0b0d10;
  --autolens-surface: #12151b;
  --autolens-surface-2: #1b2028;
  --autolens-border: #29303c;
  --autolens-accent: #6b85d9;
  --autolens-accent-dim: #3a4b7f;
  --autolens-success: #38b47f;
  --autolens-warn: #e0ac42;
  --autolens-danger: #e25859;
  --autolens-text: #e1e4ec;
  --autolens-text-dim: #929bb0;
  --autolens-text-muted: #535c72;
}
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
  background: radial-gradient(circle at 20% 0%, rgba(107,133,217,.13), transparent 30%), var(--autolens-bg) !important;
  color: var(--autolens-text) !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer, .footer, [class*="footer"], .built-with { display: none !important; }
.autolens-shell { min-height: 100vh; padding: 0; }
.autolens-header {
  min-height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: .75rem;
  padding: 0 1.5rem;
  background: rgba(18,21,27,.92);
  border-bottom: 1px solid var(--autolens-border);
  position: sticky;
  top: 0;
  z-index: 20;
  backdrop-filter: blur(12px);
}
.logo { font-size: 1.12rem; font-weight: 800; letter-spacing: .04em; }
.logo span { color: var(--autolens-accent); }
.model-badge {
  background: var(--autolens-surface-2);
  border: 1px solid var(--autolens-border);
  border-radius: 8px;
  padding: .45rem 1rem;
  font-size: .78rem;
}
.model-badge b { color: var(--autolens-accent); }
.autolens-main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 2rem 1.5rem 2.5rem;
}
.autolens-grid { gap: 1.25rem; align-items: flex-start; }
.section-label {
  font-size: .6rem;
  letter-spacing: .14em;
  color: var(--autolens-text-muted);
  padding-bottom: .45rem;
  border-bottom: 1px solid var(--autolens-border);
  margin-bottom: .8rem;
}
.panel { animation: fadeUp .45s ease-out both; }
.panel.right { animation-delay: .08s; }
#image-box, #image-box > div {
  background: var(--autolens-surface-2) !important;
  border-color: var(--autolens-border) !important;
  border-radius: 10px !important;
}
#image-box { border: 1.5px dashed var(--autolens-border) !important; overflow: hidden; }
#image-box:hover { border-color: var(--autolens-accent) !important; background: rgba(107,133,217,.06) !important; }
#classify-btn { width: 100% !important; }
#classify-btn {
  border-radius: 10px !important;
  background: linear-gradient(135deg,#6b85d9,#3a4b7f) !important;
  color: white !important;
  border: none !important;
  font-weight: 700 !important;
  letter-spacing: .08em !important;
  padding: 12px !important;
  transition: transform .15s ease, opacity .15s ease, box-shadow .15s ease !important;
}
#classify-btn:hover { transform: translateY(-1px); box-shadow: 0 10px 32px rgba(107,133,217,.15); }
.empty-card, .loading-card, .result-card, .error-card, .chart-card {
  background: var(--autolens-surface);
  border: 1px solid var(--autolens-border);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
}
.empty-card { color: var(--autolens-text-muted); text-align: center; padding: 4.5rem 1rem; }
.error-card { color: var(--autolens-danger); background: rgba(226,88,89,.07); border-color: rgba(226,88,89,.35); }
.loading-card { position: relative; height: 160px; overflow: hidden; display: flex; align-items: center; justify-content: center; color: var(--autolens-accent); letter-spacing: .12em; }
.result-card { display: flex; flex-direction: column; gap: 1rem; }
.eyebrow { font-size: .63rem; color: var(--autolens-text-muted); letter-spacing: .12em; }
.prediction-title { font-size: 1.7rem; font-weight: 800; color: var(--autolens-text); letter-spacing: .02em; }
.metric-row { display: flex; gap: .6rem; flex-wrap: wrap; }
.metric-pill { flex: 1; min-width: 140px; background: var(--autolens-surface-2); border: 1px solid var(--autolens-border); border-radius: 10px; padding: .8rem; }
.metric-pill span { display: block; font-size: 1.05rem; color: var(--autolens-accent); font-weight: 800; }
.metric-pill small { display: block; color: var(--autolens-text-muted); font-size: .62rem; margin-top: .2rem; }
#prob-chart, #prob-chart > div, #prob-chart figure { background: var(--autolens-surface) !important; border: none !important; box-shadow: none !important; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes scan { from { top: 0%; } to { top: 100%; } }
.scan-line { position: absolute; left: 0; right: 0; height: 2px; background: linear-gradient(90deg,transparent,var(--autolens-accent),transparent); animation: scan 1.35s linear infinite; box-shadow: 0 0 16px rgba(107,133,217,.55); }
.fade-up { animation: fadeUp .35s ease-out both; }
@media (max-width: 900px) {
  .autolens-header { align-items: flex-start; min-height: auto; flex-direction: column; padding: 1rem 1.5rem; }
  .autolens-main { padding: 1.25rem; }
}
"""


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="AutoLens AI") as app:
        with gr.Column(elem_classes=["autolens-shell"]):
            gr.HTML(
                """
                <header class="autolens-header">
                  <div class="logo">AutoLens <span>AI</span></div>
                </header>
                """
            )
            with gr.Column(elem_classes=["autolens-main"]):
                with gr.Row(elem_classes=["autolens-grid"]):
                    with gr.Column(scale=1, min_width=320, elem_classes=["panel"]):
                        gr.HTML("<div class='section-label'>VEHICLE IMAGE</div>")
                        image_input = gr.Image(
                            type="pil",
                            label="",
                            sources=["upload", "clipboard"],
                            height=330,
                            elem_id="image-box",
                        )
                        classify_btn = gr.Button("Classify", elem_id="classify-btn")
                        gr.HTML(
                            "<div style='color:#535c72;font-size:.68rem;text-align:center;margin-top:.3rem'>JPEG · PNG · WebP</div>"
                        )

                    with gr.Column(scale=2, min_width=420, elem_classes=["panel", "right"]):
                        gr.HTML("<div class='section-label'>PREDICTION</div>")
                        result_html = gr.HTML("<div class='empty-card'>Upload an image, then press Classify.</div>")
                        prob_chart = gr.Plot(label="", visible=False, elem_id="prob-chart")

        classify_btn.click(
            fn=classify,
            inputs=[image_input],
            outputs=[result_html, prob_chart],
        )
        image_input.upload(
            fn=reset_outputs,
            outputs=[result_html, prob_chart],
        )
        image_input.change(
            fn=reset_outputs,
            outputs=[result_html, prob_chart],
        )
        image_input.clear(
            fn=reset_outputs,
            outputs=[result_html, prob_chart],
        )
    return app


_ensure_model_downloaded()
demo = build_interface()


def main() -> None:
    def _shutdown(signum: int, _frame: Any) -> None:
        print(f"\nReceived signal {signum}, shutting down gracefully...", flush=True)
        demo.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme="base",
        css=CSS,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
