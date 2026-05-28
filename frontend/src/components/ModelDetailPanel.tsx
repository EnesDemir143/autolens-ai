// ── ModelDetailPanel — right column, collapsible, two tabs ───────────────
// HeroUI v3 Tabs: compound component via react-aria-components Tabs
import { AnimatePresence, motion } from "framer-motion";
import { Tabs } from "@heroui/react/tabs";
import { ProgressBar } from "@heroui/react/progress-bar";
import { useStore } from "../store";
import type { ModelInfo } from "../types";

function pct(v: number | null | undefined) {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function fmt(v: number | null | undefined, decimals = 4) {
  if (v == null) return "—";
  return v.toFixed(decimals);
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        gap: "0.5rem",
        padding: "0.4rem 0",
        borderBottom: "1px solid var(--autolens-border)",
        fontSize: "0.74rem",
      }}
    >
      <span style={{ color: "var(--autolens-text-muted)", flexShrink: 0 }}>{label}</span>
      <span
        style={{
          color: "var(--autolens-text-dim)",
          fontFamily: "var(--font-mono)",
          textAlign: "right",
          wordBreak: "break-all",
          maxWidth: "60%",
        }}
      >
        {value}
      </span>
    </div>
  );
}

function ConfigContent({ model }: { model: ModelInfo }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {/* 1. Architecture Section */}
      <div>
        <div style={{ 
          fontSize: "0.65rem", color: "var(--autolens-accent)", textTransform: "uppercase", 
          letterSpacing: "0.08em", marginBottom: "0.3rem", fontWeight: 700 
        }}>
          Mimari & Altyapı
        </div>
        <Row label="Base Model" value="Vision Transformer (ViT-Small)" />
        <Row label="Pre-training" value="DINOv3 (Self-Supervised)" />
        <Row label="Backend" value="ONNX Runtime CPU/MPS" />
        <Row label="Output Classes" value={String(model.num_classes)} />
      </div>

      {/* 2. Preprocessing Section */}
      <div>
        <div style={{ 
          fontSize: "0.65rem", color: "var(--autolens-accent)", textTransform: "uppercase", 
          letterSpacing: "0.08em", marginBottom: "0.3rem", fontWeight: 700 
        }}>
          Görüntü İşleme
        </div>
        <Row label="Input Resolution" value={`${model.input_size} × ${model.input_size} px`} />
        <Row label="Resize Pipeline" value={`${model.preprocessing.resize_size}px (No CenterCrop)`} />
        <Row label="Normalization" value="Custom (Data-Driven RGB)" />
      </div>

      {/* 3. Inference & Calibration Section */}
      <div>
        <div style={{ 
          fontSize: "0.65rem", color: "var(--autolens-accent)", textTransform: "uppercase", 
          letterSpacing: "0.08em", marginBottom: "0.3rem", fontWeight: 700 
        }}>
          Çıkarım (Inference)
        </div>
        <Row label="Kalibrasyon Yöntemi" value="Dirichlet (ODIR \u03bb=0.001)" />
        <Row label="Temperature (T)" value={model.temperature != null ? model.temperature.toFixed(4) : "1.0000"} />
      </div>
    </div>
  );
}

function TrainingContent({ model }: { model: ModelInfo }) {
  const t = model.training;
  const perClassEntries = Object.entries(model.per_class).sort(
    (a, b) => b[1].f1 - a[1].f1
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {/* 1. Global Metrics */}
      <div>
        <div style={{ 
          fontSize: "0.65rem", color: "var(--autolens-accent)", textTransform: "uppercase", 
          letterSpacing: "0.08em", marginBottom: "0.3rem", fontWeight: 700 
        }}>
          Test Seti Performansı
        </div>
        <Row label="Accuracy" value={pct(model.accuracy)} />
        <Row label="F1-Score (Macro)" value={pct(model.f1_macro)} />
        <Row label="F1-Score (Weighted)" value={pct(model.f1_weighted)} />
        <Row label="Validation Loss" value={fmt(model.val_loss, 4)} />
      </div>

      {/* 2. Optimization */}
      <div>
        <div style={{ 
          fontSize: "0.65rem", color: "var(--autolens-accent)", textTransform: "uppercase", 
          letterSpacing: "0.08em", marginBottom: "0.3rem", fontWeight: 700 
        }}>
          Eğitim & Optimizasyon
        </div>
        <Row label="Loss Fonksiyonu" value="Weighted Cross Entropy" />
        <Row label="Label Smoothing" value="0.05" />
        <Row label="Optimizer" value="AdamW" />
        <Row label="Learning Rate" value={t.learning_rate != null ? t.learning_rate.toExponential(2) : "1.00e-4"} />
        <Row label="Best Epoch" value={t.best_epoch != null ? String(t.best_epoch) : "29"} />
      </div>

      {/* 3. Per Class Bars */}
      {perClassEntries.length > 0 && (
        <div>
          <div style={{ 
            fontSize: "0.65rem", color: "var(--autolens-accent)", textTransform: "uppercase", 
            letterSpacing: "0.08em", marginBottom: "0.6rem", fontWeight: 700 
          }}>
            Sınıf Bazlı F1-Skor (Per-Class)
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
            {perClassEntries.map(([cls, metrics]) => (
              <div key={cls}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: "4px",
                    fontSize: "0.71rem",
                  }}
                >
                  <span style={{ color: "var(--autolens-text)", fontWeight: 500 }}>{cls}</span>
                  <span style={{ color: "var(--autolens-text-muted)", fontFamily: "var(--font-mono)" }}>
                    {pct(metrics.f1)} <span style={{fontSize: "0.65rem"}}>({metrics.support} img)</span>
                  </span>
                </div>
                <ProgressBar
                  value={metrics.f1 * 100}
                  aria-label={`${cls} f1 score`}
                  color={
                    metrics.f1 >= 0.95 ? "success" : metrics.f1 >= 0.85 ? "warning" : "danger"
                  }
                  size="sm"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function ModelDetailPanel() {
  const {
    activeModel,
    detailPanelOpen,
    detailTab,
    toggleDetailPanel,
    setDetailTab,
  } = useStore();

  if (!activeModel) return null;

  return (
    <div
      style={{
        background: "var(--autolens-surface)",
        border: "1px solid var(--autolens-border)",
        borderRadius: "12px",
        overflow: "hidden",
      }}
    >
      {/* Always-visible summary header */}
      <button
        id="detail-panel-toggle"
        onClick={toggleDetailPanel}
        aria-expanded={detailPanelOpen}
        style={{
          background: "none",
          border: "none",
          cursor: "pointer",
          padding: "1rem 1.25rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.75rem",
          borderBottom: detailPanelOpen ? "1px solid var(--autolens-border)" : "none",
          width: "100%",
          textAlign: "left",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
          <span
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "0.88rem",
              fontWeight: 700,
              color: "var(--autolens-text)",
            }}
          >
            {activeModel.model_name}
          </span>
          <span
            style={{
              fontSize: "0.7rem",
              color: "var(--autolens-text-muted)",
              fontFamily: "var(--font-mono)",
            }}
          >
            acc {pct(activeModel.accuracy)} · F1 {pct(activeModel.f1_macro)}
          </span>
        </div>
        <svg
          width="14" height="14" viewBox="0 0 14 14" fill="none"
          stroke="currentColor" strokeWidth="1.5"
          style={{
            color: "var(--autolens-text-muted)",
            transform: detailPanelOpen ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform 0.25s ease",
            flexShrink: 0,
          }}
        >
          <polyline points="2 5 7 10 12 5"/>
        </svg>
      </button>

      {/* Expandable detail body */}
      <AnimatePresence initial={false}>
        {detailPanelOpen && (
          <motion.div
            key="detail-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            style={{ overflow: "hidden" }}
          >
            {/* Cross-fade when active model changes */}
            <AnimatePresence mode="wait">
              <motion.div
                key={activeModel.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.18 }}
                style={{
                  padding: "0.75rem 1rem 1rem",
                  maxHeight: "calc(100vh - 180px)",
                  overflowY: "auto",
                }}
              >
                {/* HeroUI v3 Tabs compound component */}
                <Tabs.Root
                  selectedKey={detailTab}
                  onSelectionChange={(key) => setDetailTab(key as "config" | "training")}
                >
                  <Tabs.List aria-label="Model detail sections">
                    <Tabs.Tab id="config">Config</Tabs.Tab>
                    <Tabs.Tab id="training">Training</Tabs.Tab>
                  </Tabs.List>
                  <Tabs.Panel id="config" style={{ paddingTop: "0.75rem" }}>
                    <ConfigContent model={activeModel} />
                  </Tabs.Panel>
                  <Tabs.Panel id="training" style={{ paddingTop: "0.75rem" }}>
                    <TrainingContent model={activeModel} />
                  </Tabs.Panel>
                </Tabs.Root>
              </motion.div>
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
