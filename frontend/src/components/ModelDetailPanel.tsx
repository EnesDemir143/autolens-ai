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
    <div style={{ display: "flex", flexDirection: "column" }}>
      <Row label="Model name" value={model.model_name} />
      <Row label="Input size" value={`${model.input_size}px`} />
      <Row
        label="Resize / Crop"
        value={`${model.preprocessing.resize_size}px / ${model.preprocessing.crop_size}px`}
      />
      <Row label="Mean" value={model.preprocessing.mean.map((v) => v.toFixed(4)).join(", ")} />
      <Row label="Std" value={model.preprocessing.std.map((v) => v.toFixed(4)).join(", ")} />
      <Row label="Temperature (T)" value={model.temperature != null ? model.temperature.toFixed(4) : "—"} />
      <Row label="Backend" value="ONNX Runtime" />
      <Row label="Run ID" value={model.source_run_id} />
    </div>
  );
}

function TrainingContent({ model }: { model: ModelInfo }) {
  const t = model.training;
  const perClassEntries = Object.entries(model.per_class).sort(
    (a, b) => b[1].accuracy - a[1].accuracy
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.1rem" }}>
      <Row label="Best epoch" value={t.best_epoch != null ? String(t.best_epoch) : "—"} />
      <Row label="Total epochs" value={t.total_epochs_run != null ? String(t.total_epochs_run) : "—"} />
      <Row label="Learning rate" value={t.learning_rate != null ? t.learning_rate.toExponential(2) : "—"} />
      <Row label="Weight decay" value={t.weight_decay != null ? t.weight_decay.toExponential(2) : "—"} />
      <Row label="Val loss (test)" value={fmt(model.val_loss, 4)} />
      <Row label="Accuracy" value={pct(model.accuracy)} />
      <Row label="F1 macro" value={pct(model.f1_macro)} />
      <Row label="F1 weighted" value={pct(model.f1_weighted)} />

      {perClassEntries.length > 0 && (
        <div style={{ marginTop: "0.75rem" }}>
          <div
            style={{
              fontSize: "0.63rem",
              color: "var(--autolens-text-muted)",
              letterSpacing: "0.12em",
              marginBottom: "0.5rem",
            }}
          >
            PER-CLASS ACCURACY
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
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
                  <span style={{ color: "var(--autolens-text-dim)" }}>{cls}</span>
                  <span style={{ color: "var(--autolens-text-muted)", fontFamily: "var(--font-mono)" }}>
                    {pct(metrics.accuracy)} ({metrics.support})
                  </span>
                </div>
                <ProgressBar
                  value={metrics.accuracy * 100}
                  aria-label={`${cls} accuracy`}
                  color={
                    metrics.accuracy >= 0.9 ? "success" : metrics.accuracy >= 0.75 ? "warning" : "danger"
                  }
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
