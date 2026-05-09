#!/usr/bin/env python
"""AutoLens AI Gradio demo — HF Spaces compatible, model-agnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from autolens_ai.inference import ONNXPredictor

PREDICTOR: ONNXPredictor | None = None


def _load_predictor() -> ONNXPredictor:
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = ONNXPredictor.from_default_config()
    return PREDICTOR


def _build_prob_chart(probabilities: dict[str, float]) -> plt.Figure:
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in sorted_items]
    probs = [i[1] for i in sorted_items]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    colors = ["#6366f1" if i == 0 else "#1e293b" for i in range(len(labels))]
    bars = ax.barh(range(len(labels)), probs, color=colors, height=0.55, edgecolor="none")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color="#cbd5e1")
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors="#374151", labelsize=8)
    ax.tick_params(axis="y", length=0)

    for bar, prob in zip(bars, probs):
        color = "#a5b4fc" if prob == max(probs) else "#6b7280"
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=8.5, color=color,
                fontweight="bold" if prob == max(probs) else "normal")

    plt.tight_layout(pad=0.4)
    return fig


def classify(image: Image.Image | None) -> tuple[str, Any]:
    if image is None:
        return ("<div style='color:#4b5563;text-align:center;padding:60px 0;font-size:14px'>Upload a vehicle image to begin</div>", gr.update(visible=False))
    try:
        predictor = _load_predictor()
        result = predictor.predict(image)
        pred = result["predicted_display"]
        conf = result["confidence"]
        latency = result["latency_ms"]
        chart = _build_prob_chart(result["probabilities"])

        conf_color = "#4ade80" if conf >= 0.85 else ("#fb923c" if conf >= 0.60 else "#f87171")
        conf_tag = "High" if conf >= 0.85 else ("Moderate" if conf >= 0.60 else "Low")

        html = f"""<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
          <div style="background:linear-gradient(135deg,#1e1b4b,#1e293b);border:1px solid #4338ca;border-radius:12px;padding:18px 22px;margin-bottom:12px">
            <div style="color:#818cf8;font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase">Predicted Class</div>
            <div style="color:#f1f5f9;font-size:2rem;font-weight:800;margin-top:4px">{pred}</div>
          </div>
          <div style="display:flex;gap:10px">
            <div style="flex:1;background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;text-align:center">
              <div style="color:#6b7280;font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase">Confidence</div>
              <div style="color:{conf_color};font-size:1.2rem;font-weight:700;margin-top:3px">{conf:.1%}</div>
              <div style="color:#4b5563;font-size:10px">{conf_tag}</div>
            </div>
            <div style="flex:1;background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;text-align:center">
              <div style="color:#6b7280;font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase">Latency</div>
              <div style="color:#a5b4fc;font-size:1.2rem;font-weight:700;margin-top:3px">{latency:.0f} ms</div>
              <div style="color:#4b5563;font-size:10px">ONNX Runtime</div>
            </div>
          </div>
        </div>"""
        return (html, gr.update(value=chart, visible=True))
    except Exception as exc:
        return (f"<div style='color:#f87171;padding:12px;background:#1e293b;border-radius:8px'>Error: {exc}</div>", gr.update(visible=False))


def on_start(image):
    if image is None:
        return ("<div style='color:#4b5563;text-align:center;padding:60px 0;font-size:14px'>⚠️ Please upload an image first.</div>", gr.update(visible=False))
    return ("<div style='color:#94a3b8;text-align:center;padding:60px 0;font-size:14px'>⏳ Analyzing…</div>", gr.update(visible=False))


CSS = """
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: #0a0a14 !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer, .footer, [class*="footer"], .built-with { display: none !important; }
.al-header {
    background: #0f0f1f; border-bottom: 1px solid #1e293b;
    padding: 18px 32px;
}
.al-header h1 {
    font-size: 1.5rem; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.al-header p { color: #4b5563; font-size: 12px; margin: 4px 0 0; }
.al-left { background: #0f0f1f; border-right: 1px solid #1e293b; padding: 20px; }
.classify-btn > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important; border-radius: 10px !important;
    color: white !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 13px !important;
    width: 100% !important; margin-top: 12px !important;
}
.classify-btn > button:hover { opacity: 0.85 !important; }
.gr-plot, .gr-plot > div, .gr-plot figure { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
.gr-plot label { display: none !important; }
[data-testid="plot"] { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
[data-testid="plot"] > div { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="plot"] label, [data-testid="plot"] .label-wrap { display: none !important; }
#prob-chart, #prob-chart > div, #prob-chart > div > div { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
#prob-chart label, #prob-chart .label-wrap { display: none !important; }
"""


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="AutoLens AI") as demo:
        gr.HTML("""
        <div class="al-header">
          <h1>AutoLens AI</h1>
          <p>8-class vehicle body type classifier · EfficientNet-B2 · ONNX Runtime</p>
        </div>
        """)

        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=280, elem_classes=["al-left"]):
                image_input = gr.Image(
                    type="pil", label="Vehicle Image",
                    sources=["upload", "clipboard"], height=300,
                )
                classify_btn = gr.Button("Classify →", variant="primary", elem_classes=["classify-btn"])
                gr.HTML("<div style='color:#374151;font-size:10px;margin-top:8px;text-align:center'>JPG · PNG · WebP</div>")

            with gr.Column(scale=3, min_width=380):
                result_html = gr.HTML(
                    value="<div style='color:#4b5563;text-align:center;padding:80px 0;font-size:14px'>Upload a vehicle image to begin</div>"
                )
                prob_chart = gr.Plot(label="", visible=False, elem_id="prob-chart")

        classify_btn.click(
            fn=on_start, inputs=[image_input], outputs=[result_html, prob_chart],
        ).then(
            fn=classify, inputs=[image_input], outputs=[result_html, prob_chart],
        )
        image_input.upload(
            fn=on_start, inputs=[image_input], outputs=[result_html, prob_chart],
        ).then(
            fn=classify, inputs=[image_input], outputs=[result_html, prob_chart],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,
        theme=gr.themes.Base(),
    )


if __name__ == "__main__":
    main()
