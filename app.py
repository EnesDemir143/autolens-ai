#!/usr/bin/env python
"""AutoLens AI Gradio demo — HF Spaces compatible."""

from __future__ import annotations

import io
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
PYTORCH_MODEL: Any | None = None


def _load_predictor() -> ONNXPredictor:
    global PREDICTOR
    if PREDICTOR is None:
        if not Path("artifacts/demo/active_model.json").exists():
            # HF Spaces / fresh clone fallback: download flagship artifacts on first use.
            from scripts.download_from_hf import download_model, write_active_model

            export_dir = download_model("dinov3-weighted", force=False)
            write_active_model(export_dir)
        PREDICTOR = ONNXPredictor.from_default_config()
    return PREDICTOR


def _load_pytorch_model() -> Any:
    """Load safetensors model lazily for Grad-CAM."""
    global PYTORCH_MODEL
    if PYTORCH_MODEL is None:
        if not Path("artifacts/demo/active_model.json").exists():
            _load_predictor()
        import json

        from autolens_ai.inference.artifact import load_model_from_artifact

        active = json.loads(Path("artifacts/demo/active_model.json").read_text(encoding="utf-8"))
        model_path = Path(active["model_path"])
        export_dir = model_path.parent
        safetensors_path = export_dir / "model.safetensors"
        metadata_path = Path(active["metadata_path"])
        if not safetensors_path.exists():
            raise FileNotFoundError(
                "model.safetensors was not downloaded. Rebuild the Space after the updated "
                "download script is pushed."
            )
        PYTORCH_MODEL = load_model_from_artifact(safetensors_path, metadata_path, map_location="cpu")
        PYTORCH_MODEL.eval()
    return PYTORCH_MODEL


def _build_prob_chart(probabilities: dict[str, float]) -> plt.Figure:
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in sorted_items]
    probs = [i[1] for i in sorted_items]
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

    for idx, (bar, prob) in enumerate(zip(bars, probs)):
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


def classify(image: Image.Image | None) -> tuple[str, Any, Any, Any, Any]:
    if image is None:
        return (
            "<div class='empty-card'>Select a vehicle image first.</div>",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    try:
        predictor = _load_predictor()
        result = predictor.predict(image)
        chart = _build_prob_chart(result["probabilities"])
        return (
            _result_card(result),
            gr.update(value=chart, visible=True),
            gr.update(visible=True),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )
    except Exception as exc:
        return (
            f"<div class='error-card'>Error: {exc}</div>",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def reset_outputs() -> tuple[str, Any, Any, Any, Any]:
    """Hide stale prediction outputs when image changes or is cleared."""
    return (
        "<div class='empty-card'>Upload an image, then press Classify.</div>",
        gr.update(value=None, visible=False),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
    )


def start_loading(image: Image.Image | None) -> tuple[str, Any, Any, Any, Any]:
    if image is None:
        return reset_outputs()
    return (
        "<div class='loading-card'><div class='scan-line'></div><span>ANALYZING…</span></div>",
        gr.update(value=None, visible=False),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
    )


def _target_layers_for_gradcam(model: Any) -> list[Any]:
    import torch

    # For timm ViT/DINOv3, patch_embed.proj is a Conv2d and gives Grad-CAM a native
    # 4D activation map. This is more robust on Spaces than token-layer reshape hooks.
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        return [model.patch_embed.proj]

    last_conv: torch.nn.Module | None = None
    for mod in model.modules():
        if isinstance(mod, torch.nn.Conv2d):
            last_conv = mod
    if last_conv is None:
        raise ValueError("Could not find a Grad-CAM target layer for this model.")
    return [last_conv]


def generate_gradcam(image: Image.Image | None) -> tuple[Any, Any]:
    if image is None:
        return gr.update(visible=False), gr.update(value="<div class='error-card'>Upload an image first.</div>", visible=True)
    try:
        import json

        import numpy as np
        import torch
        from pytorch_grad_cam import GradCAM  # type: ignore[import-untyped]
        from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore[import-untyped]

        model = _load_pytorch_model()
        active = json.loads(Path("artifacts/demo/active_model.json").read_text(encoding="utf-8"))
        meta = json.loads(Path(active["metadata_path"]).read_text(encoding="utf-8"))
        pre = meta["preprocessing"]
        size = pre.get("image_size", 224)
        resize_size = pre.get("resize_size", size + 32)
        crop_size = pre.get("crop_size", size)
        mean = np.array(pre.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(pre.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

        if image.mode != "RGB":
            image = image.convert("RGB")
        img_resized = image.resize((resize_size, resize_size), Image.Resampling.BILINEAR)
        left = (resize_size - crop_size) // 2
        top = (resize_size - crop_size) // 2
        img_cropped = img_resized.crop((left, top, left + crop_size, top + crop_size))

        arr = np.array(img_cropped).astype(np.float32) / 255.0
        arr_normalized = (arr - mean) / std
        input_tensor = torch.from_numpy(np.transpose(arr_normalized, (2, 0, 1))).unsqueeze(0)
        target_layers = _target_layers_for_gradcam(model)

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

        visualization = show_cam_on_image(arr, grayscale_cam, use_rgb=True)
        overlay = Image.fromarray(visualization)
        w, h = img_cropped.size
        combined = Image.new("RGB", (w * 2, h), color=(11, 13, 16))
        combined.paste(img_cropped, (0, 0))
        combined.paste(overlay, (w, 0))
        return gr.update(value=combined, visible=True), gr.update(value="", visible=False)
    except Exception as exc:
        return gr.update(visible=False), gr.update(value=f"<div class='error-card'>Grad-CAM error: {exc}</div>", visible=True)


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
.autolens-shell { min-height: 100vh; }
.autolens-header {
  height: 60px; display: flex; align-items: center; justify-content: space-between;
  padding: 0 1.5rem; background: rgba(18,21,27,.92); border-bottom: 1px solid var(--autolens-border);
  position: sticky; top: 0; z-index: 20; backdrop-filter: blur(12px);
}
.logo { font-size: 1.12rem; font-weight: 800; letter-spacing: .04em; }
.logo span { color: var(--autolens-accent); }
.model-badge { background: var(--autolens-surface-2); border:1px solid var(--autolens-border); border-radius:8px; padding:.45rem 1rem; font-size:.78rem; }
.model-badge b { color: var(--autolens-accent); }
.autolens-main { padding: 2rem 1.5rem 2.5rem; }
.autolens-grid { display:grid; grid-template-columns: 340px minmax(420px, 720px); gap:1.25rem; max-width:1100px; margin:0 auto; align-items:start; }
.section-label { font-size:.6rem; letter-spacing:.14em; color:var(--autolens-text-muted); padding-bottom:.45rem; border-bottom:1px solid var(--autolens-border); margin-bottom:.8rem; }
.panel { animation: fadeUp .45s ease-out both; }
.panel.right { animation-delay:.08s; }
#image-box, #image-box > div { background: var(--autolens-surface-2) !important; border-color: var(--autolens-border) !important; border-radius: 10px !important; }
#image-box { border: 1.5px dashed var(--autolens-border) !important; overflow: hidden; }
#image-box:hover { border-color: var(--autolens-accent) !important; background: rgba(107,133,217,.06) !important; }
#classify-btn button, #gradcam-btn button {
  width:100% !important; border-radius:10px !important; background:linear-gradient(135deg,#6b85d9,#3a4b7f) !important;
  color:white !important; border:none !important; font-weight:700 !important; letter-spacing:.08em !important; padding: 12px !important;
  transition: transform .15s ease, opacity .15s ease, box-shadow .15s ease !important;
}
#gradcam-btn button { background: transparent !important; border: 1px solid var(--autolens-border) !important; color: var(--autolens-text-dim) !important; }
#classify-btn button:hover, #gradcam-btn button:hover { transform: translateY(-1px); box-shadow: 0 10px 32px rgba(107,133,217,.15); }
.empty-card, .loading-card, .result-card, .error-card, .chart-card {
  background: var(--autolens-surface); border:1px solid var(--autolens-border); border-radius:12px; padding:1.25rem 1.5rem;
}
.empty-card { color: var(--autolens-text-muted); text-align:center; padding: 4.5rem 1rem; }
.error-card { color: var(--autolens-danger); background: rgba(226,88,89,.07); border-color: rgba(226,88,89,.35); }
.loading-card { position:relative; height: 160px; overflow:hidden; display:flex; align-items:center; justify-content:center; color:var(--autolens-accent); letter-spacing:.12em; }
.result-card { display:flex; flex-direction:column; gap:1rem; }
.eyebrow { font-size:.63rem; color:var(--autolens-text-muted); letter-spacing:.12em; }
.prediction-title { font-size:1.7rem; font-weight:800; color:var(--autolens-text); letter-spacing:.02em; }
.metric-row { display:flex; gap:.6rem; flex-wrap:wrap; }
.metric-pill { flex:1; min-width:140px; background:var(--autolens-surface-2); border:1px solid var(--autolens-border); border-radius:10px; padding:.8rem; }
.metric-pill span { display:block; font-size:1.05rem; color:var(--autolens-accent); font-weight:800; }
.metric-pill small { display:block; color:var(--autolens-text-muted); font-size:.62rem; margin-top:.2rem; }
#prob-chart, #prob-chart > div, #prob-chart figure { background: var(--autolens-surface) !important; border: none !important; box-shadow: none !important; }
#gradcam-output, #gradcam-output > div { background: var(--autolens-surface) !important; border-color: var(--autolens-border) !important; border-radius:12px !important; overflow:hidden; }
@keyframes fadeUp { from { opacity:0; transform: translateY(12px); } to { opacity:1; transform: translateY(0); } }
@keyframes scan { from { top:0%; } to { top:100%; } }
.scan-line { position:absolute; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,var(--autolens-accent),transparent); animation: scan 1.35s linear infinite; box-shadow:0 0 16px rgba(107,133,217,.55); }
.fade-up { animation: fadeUp .35s ease-out both; }
@media (max-width: 900px) { .autolens-grid { grid-template-columns: 1fr; } .autolens-header { align-items:flex-start; height:auto; gap:.7rem; flex-direction:column; padding:1rem 1.5rem; } }
"""


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="AutoLens AI") as demo:
        gr.HTML(
            """
            <div class="autolens-shell">
              <header class="autolens-header">
                <div class="logo">AutoLens <span>AI</span></div>
                <div class="model-badge"><b>●</b> AutoLens ViT-S <span style="color:#535c72;margin-left:.5rem">acc 96.0%</span></div>
              </header>
              <main class="autolens-main"><div class="autolens-grid">
            """
        )
        with gr.Column(elem_classes=["panel"]):
            gr.HTML("<div class='section-label'>VEHICLE IMAGE</div>")
            image_input = gr.Image(
                type="pil",
                label="",
                sources=["upload", "clipboard"],
                height=330,
                elem_id="image-box",
            )
            classify_btn = gr.Button("Classify", elem_id="classify-btn")
            gr.HTML("<div style='color:#535c72;font-size:.68rem;text-align:center;margin-top:.3rem'>JPEG · PNG · WebP</div>")

        with gr.Column(elem_classes=["panel", "right"]):
            gr.HTML("<div class='section-label'>PREDICTION</div>")
            result_html = gr.HTML("<div class='empty-card'>Upload an image, then press Classify.</div>")
            prob_chart = gr.Plot(label="", visible=False, elem_id="prob-chart")
            gradcam_btn = gr.Button("Show GradCAM", visible=False, elem_id="gradcam-btn")
            gradcam_status = gr.HTML("", visible=False)
            gradcam_output = gr.Image(label="Original | Grad-CAM Overlay", visible=False, elem_id="gradcam-output")

        gr.HTML("</div></main></div>")

        classify_btn.click(
            fn=start_loading,
            inputs=[image_input],
            outputs=[result_html, prob_chart, gradcam_btn, gradcam_output, gradcam_status],
        ).then(
            fn=classify,
            inputs=[image_input],
            outputs=[result_html, prob_chart, gradcam_btn, gradcam_output, gradcam_status],
        )
        image_input.upload(
            fn=reset_outputs,
            outputs=[result_html, prob_chart, gradcam_btn, gradcam_output, gradcam_status],
        )
        image_input.change(
            fn=reset_outputs,
            outputs=[result_html, prob_chart, gradcam_btn, gradcam_output, gradcam_status],
        )
        image_input.clear(
            fn=reset_outputs,
            outputs=[result_html, prob_chart, gradcam_btn, gradcam_output, gradcam_status],
        )
        gradcam_btn.click(
            fn=generate_gradcam,
            inputs=[image_input],
            outputs=[gradcam_output, gradcam_status],
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
