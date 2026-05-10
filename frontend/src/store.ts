// ── Zustand global store ────────────────────────────────────────────────────
import { create } from "zustand";
import * as api from "./api";
import type {
  DetailTab,
  ModelInfo,
  PredictionResult,
  PredictionStatus,
} from "./types";

interface Store {
  // Model state
  models: ModelInfo[];
  activeModel: ModelInfo | null;
  isModelSwitching: boolean;
  fetchModels: () => Promise<void>;
  setActiveModel: (id: string) => Promise<void>;

  // Single-image prediction state
  status: PredictionStatus;
  result: PredictionResult | null;
  error: string | null;
  selectedFile: File | null;
  setFile: (file: File | null) => void;
  predict: () => Promise<void>;
  resetPrediction: () => void;

  // GradCAM state
  gradcam: string | null;
  isGeneratingGradcam: boolean;
  gradcamVisible: boolean;
  fetchGradcam: () => Promise<void>;
  toggleGradcam: () => void;

  // UI state
  detailPanelOpen: boolean;
  detailTab: DetailTab;
  toggleDetailPanel: () => void;
  setDetailTab: (tab: DetailTab) => void;
}

export const useStore = create<Store>((set, get) => ({
  // ── Model ─────────────────────────────────────────────────────────────
  models: [],
  activeModel: null,
  isModelSwitching: false,

  fetchModels: async () => {
    const models = await api.fetchModels();
    const active = models.find((m) => m.is_active) ?? models[0] ?? null;
    set({ models, activeModel: active });
  },

  setActiveModel: async (id: string) => {
    set({ isModelSwitching: true });
    try {
      await api.setActiveModel(id);
      const models = await api.fetchModels();
      const active = models.find((m) => m.is_active) ?? null;
      set({
        models,
        activeModel: active,
        status: "idle",
        result: null,
        error: null,
        gradcam: null,
        gradcamVisible: false,
      });
    } finally {
      set({ isModelSwitching: false });
    }
  },

  // ── Prediction ────────────────────────────────────────────────────────
  status: "idle",
  result: null,
  error: null,
  selectedFile: null,

  setFile: (file) => {
    set({
      selectedFile: file,
      status: "idle",
      result: null,
      error: null,
      gradcam: null,
      gradcamVisible: false,
    });
  },

  predict: async () => {
    const { selectedFile } = get();
    if (!selectedFile) return;
    set({ status: "analyzing", result: null, error: null, gradcam: null, gradcamVisible: false });
    try {
      const result = await api.predict(selectedFile);
      set({ status: "done", result });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Prediction failed. Try again.";
      set({ status: "error", error: msg });
    }
  },

  resetPrediction: () => {
    set({ status: "idle", result: null, error: null, gradcam: null, gradcamVisible: false });
  },

  // ── GradCAM ──────────────────────────────────────────────────────────
  gradcam: null,
  isGeneratingGradcam: false,
  gradcamVisible: false,

  fetchGradcam: async () => {
    const { selectedFile } = get();
    if (!selectedFile) return;
    set({ isGeneratingGradcam: true });
    try {
      const image = await api.fetchGradcam(selectedFile);
      set({ gradcam: image, gradcamVisible: true });
    } catch {
      set({ gradcam: null, gradcamVisible: false });
    } finally {
      set({ isGeneratingGradcam: false });
    }
  },

  toggleGradcam: () => {
    const { gradcamVisible, gradcam, selectedFile } = get();
    if (gradcamVisible) {
      set({ gradcamVisible: false });
    } else if (gradcam) {
      set({ gradcamVisible: true });
    } else if (selectedFile) {
      get().fetchGradcam();
    }
  },

  // ── UI ────────────────────────────────────────────────────────────────
  detailPanelOpen: false,
  detailTab: "config",

  toggleDetailPanel: () => set((s) => ({ detailPanelOpen: !s.detailPanelOpen })),
  setDetailTab: (tab) => set({ detailTab: tab }),
}));
