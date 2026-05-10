// ── Shared TypeScript types for AutoLens AI demo UI ──────────────────────

export interface PerClassMetrics {
  accuracy: number;
  support: number;
}

export interface TrainingInfo {
  best_epoch: number | null;
  total_epochs_run: number | null;
  learning_rate: number | null;
  weight_decay: number | null;
}

export interface Preprocessing {
  image_size: number;
  resize_size: number;
  crop_size: number;
  mean: number[];
  std: number[];
  color_mode: string;
}

export interface ModelInfo {
  id: string;
  model_name: string;
  accuracy: number | null;
  f1_macro: number | null;
  f1_weighted: number | null;
  val_loss: number | null;
  temperature: number | null;
  num_classes: number;
  input_size: number;
  pretrained: boolean;
  source_run_id: string;
  is_active: boolean;
  per_class: Record<string, PerClassMetrics>;
  training: TrainingInfo;
  preprocessing: Preprocessing;
}

export interface PredictionResult {
  predicted_class: string;
  predicted_display: string;
  confidence: number;
  probabilities: Record<string, number>;
  latency_ms: number;
  temperature: number;
}

export type PredictionStatus = "idle" | "analyzing" | "done" | "error";
export type DetailTab = "config" | "training";
