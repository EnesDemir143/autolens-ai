// ── Typed API client ──────────────────────────────────────────────────────
import axios from "axios";
import type { ModelInfo, PredictionResult } from "./types";

const http = axios.create({ baseURL: "/" });

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await http.get<ModelInfo[]>("/api/models", {
    headers: { "Cache-Control": "no-store" },
  });
  return res.data;
}

export async function setActiveModel(id: string): Promise<void> {
  await http.post("/api/models/active", { id });
}

export async function predict(file: File): Promise<PredictionResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await http.post<PredictionResult>("/api/predict", form);
  return res.data;
}

export async function fetchGradcam(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  const res = await http.post<{ image: string }>("/api/gradcam", form);
  return res.data.image;
}
