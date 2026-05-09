#!/usr/bin/env python
"""AutoLens AI Gradio demo.

Loads the active ONNX artifact config and serves a web UI for car body type
classification. Model-agnostic: replace artifacts/demo/active_model.json to
swap the backend without touching this file.
"""

from __future__ import annotations

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
    """Lazy-load the global predictor."""
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = ONNXPredictor.from_default_config()
    return PREDICTOR


def _build_prob_chart(probabilities: dict[str, float]) -> plt.Figure:
    """Build a horizontal bar chart of class probabilities."""
    labels = list(probabilities.keys())
    probs = list(probabilities.values())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    y_pos = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

    bars = ax.barh(y_pos, probs, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=9)
    ax.set_title("Class Probability Distribution", fontsize=10, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    return fig


def classify(image: Image.Image | None) -> tuple[Any, ...]:
    """Run classification and return all UI outputs.

    Returns a tuple matching the Gradio output component order:
    1. predicted_class (text)
    2. confidence (text)
    3. latency (text)
    4. prob_chart (matplotlib figure)
    5. status (text)
    """
    if image is None:
        return (
            "—",
            "—",
            "—",
            None,
            gr.update(visible=False),
        )

    try:
        predictor = _load_predictor()
        result = predictor.predict(image)

        pred_text = f"{result['predicted_display']}"
        conf_text = f"{result['confidence']:.1%}"
        latency_text = f"{result['latency_ms']:.0f} ms"
        chart = _build_prob_chart(result["probabilities"])

        return (
            pred_text,
            conf_text,
            latency_text,
            chart,
            gr.update(visible=True),
        )
    except Exception as exc:
        return (
            "Error",
            str(exc),
            "—",
            None,
            gr.update(visible=False),
        )


def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks UI."""
    with gr.Blocks(title="AutoLens AI") as demo:
        gr.HTML("""
            <div class="hero">
                <h1>AutoLens AI</h1>
                <p>Upload a vehicle image to classify its body type</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Vehicle Image",
                    sources=["upload", "clipboard"],
                )
                classify_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                # Loading / status
                status_text = gr.Markdown(
                    value="",
                    visible=False,
                )

                # Results container
                with gr.Column(visible=False) as result_col:
                    gr.Markdown("### Prediction")
                    predicted_class = gr.Textbox(
                        label="Predicted Class",
                        interactive=False,
                    )
                    confidence = gr.Textbox(
                        label="Confidence",
                        interactive=False,
                    )
                    latency = gr.Textbox(
                        label="Inference Latency",
                        interactive=False,
                    )
                    gr.Markdown("### Probability Distribution")
                    prob_chart = gr.Plot(label="Class Probabilities")

        # Event wiring
        def on_classify_click(image):
            if image is None:
                return (
                    gr.update(visible=True, value="⚠️ Please upload an image first."),
                    gr.update(visible=False),
                    "—", "—", "—", None,
                )
            # Show loading, hide results
            return (
                gr.update(visible=True, value="⏳ Analyzing vehicle image…"),
                gr.update(visible=False),
                "—", "—", "—", None,
            )

        def on_classify_done(image):
            return classify(image)

        classify_btn.click(
            fn=on_classify_click,
            inputs=[image_input],
            outputs=[
                status_text,
                result_col,
                predicted_class,
                confidence,
                latency,
                prob_chart,
            ],
        ).then(
            fn=on_classify_done,
            inputs=[image_input],
            outputs=[
                status_text,
                result_col,
                predicted_class,
                confidence,
                latency,
                prob_chart,
            ],
        )

        # Also trigger on image upload (auto-classify)
        image_input.upload(
            fn=on_classify_click,
            inputs=[image_input],
            outputs=[
                status_text,
                result_col,
                predicted_class,
                confidence,
                latency,
                prob_chart,
            ],
        ).then(
            fn=on_classify_done,
            inputs=[image_input],
            outputs=[
                status_text,
                result_col,
                predicted_class,
                confidence,
                latency,
                prob_chart,
            ],
        )

    return demo


def main() -> None:
    """Launch the Gradio demo."""
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css="""
        .hero { text-align: center; margin-bottom: 1rem; }
        .hero h1 { margin-bottom: 0.2rem; }
        .hero p { color: #666; font-size: 0.95rem; }
        .result-box { padding: 1rem; border-radius: 8px; background: #f8f9fa; }
        .loading { text-align: center; color: #555; font-style: italic; padding: 2rem; }
    """,
    )


if __name__ == "__main__":
    main()
