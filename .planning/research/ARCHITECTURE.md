# Architecture Research

## Components

1. Environment and configuration layer: uv, pyproject, Hydra configs, dotenv for credentials.
2. Data acquisition layer: Kaggle CLI, Hugging Face datasets/hub, source manifests.
3. Dataset curation layer: class mapping, deduplication/manual review hooks, split creation, imbalance report.
4. Training layer: Lightning datamodule, model factory, training module, callbacks, checkpointing.
5. Evaluation layer: metric computation, classification report, plots, confusion matrix, artifact size check.
6. Inference layer: saved model loader, preprocessing parity, prediction API.
7. UI layer: Gradio Blocks interface with image upload and probability visualization.
8. Reporting layer: final IEEE LaTeX report generated after implementation evidence exists.

## Data Flow

Public sources → raw dataset folders → curated 8-class dataset → train/validation split → model experiments → metric/plot artifacts → final model selection → Gradio inference → report evidence.

## Build Order

1. Initialize project/tooling.
2. Acquire and curate dataset.
3. Build baseline train/eval pipeline.
4. Run model comparison.
5. Package inference and UI.
6. Generate report and final submission artifacts.
