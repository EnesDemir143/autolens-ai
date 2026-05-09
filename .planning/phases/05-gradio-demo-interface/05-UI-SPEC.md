# Phase 5 UI-SPEC: Gradio Demo Interface
**Branch:** `feat/gradio-demo-interface`

## Visual Direction

Clean modern demo page using Gradio Blocks: compact hero title, short explanation, two-column body, image upload/preview on the left, result and class probability chart on the right.

## Required UI Elements

- Drag-and-drop image upload.
- Uploaded image preview.
- Clear classify button.
- Prominent predicted class text.
- Confidence score.
- Probability distribution for exactly 8 classes.
- Latency/status note for demo confidence.
- Loading/progress state during prediction, e.g. “Analyzing vehicle image…” with spinner/status feedback until results appear.

## Acceptance Contract

UI passes only if every assignment interface requirement is visible without opening developer tools or reading logs.


## Interaction State Requirement

During inference, especially for DINOv3 ONNX CPU predictions, the UI must show a clear loading/progress/status state and must not look frozen. Results replace the loading state only after prediction returns.
