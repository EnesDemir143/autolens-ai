// ── ResultCard — slide-in, confidence chip, error state ──────────────────
import { AnimatePresence, motion } from "framer-motion";
import { Button } from "@heroui/react/button";
import { Chip } from "@heroui/react/chip";
import { useStore } from "../store";

function confidenceColor(conf: number): "success" | "warning" | "danger" {
  if (conf >= 0.85) return "success";
  if (conf >= 0.6) return "warning";
  return "danger";
}

export function ResultCard() {
  const { status, result, error, resetPrediction, setFile } = useStore();

  return (
    <AnimatePresence mode="wait">
      {status === "done" && result && (
        <motion.div
          key="result"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -16 }}
          transition={{ duration: 0.35, ease: "easeOut" }}
          style={{
            background: "var(--autolens-surface)",
            border: "1px solid var(--autolens-border)",
            borderRadius: "12px",
            padding: "1.25rem 1.5rem",
            display: "flex",
            flexDirection: "column",
            gap: "1rem",
          }}
        >
          {/* Predicted class */}
          <div>
            <div
              style={{
                fontSize: "0.63rem",
                color: "var(--autolens-text-muted)",
                letterSpacing: "0.12em",
                marginBottom: "0.4rem",
              }}
            >
              PREDICTED CLASS
            </div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "1.55rem",
                fontWeight: 500,
                color: "var(--autolens-text)",
                letterSpacing: "0.02em",
                lineHeight: 1.15,
              }}
            >
              {result.predicted_display}
            </div>
          </div>

          {/* Chips row */}
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <Chip size="sm" color={confidenceColor(result.confidence)} variant="soft">
              {(result.confidence * 100).toFixed(1)}% confidence
            </Chip>
            <Chip size="sm" color="accent" variant="soft">
              {result.latency_ms.toFixed(1)} ms
            </Chip>
          </div>

          {/* Reset */}
          <button
            id="new-image-btn"
            onClick={() => setFile(null)}
            style={{
              alignSelf: "flex-end",
              background: "none",
              border: "none",
              cursor: "pointer",
              color: "var(--autolens-text-muted)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.72rem",
              letterSpacing: "0.05em",
              padding: "0",
              transition: "color 0.15s",
            }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.color = "var(--autolens-text-dim)")}
            onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.color = "var(--autolens-text-muted)")}
          >
            ↺ new image
          </button>
        </motion.div>
      )}

      {status === "error" && (
        <motion.div
          key="error"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          style={{
            background: "rgba(248,113,113,0.06)",
            border: "1px solid rgba(248,113,113,0.3)",
            borderRadius: "12px",
            padding: "1.25rem 1.5rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.75rem",
          }}
        >
          <div
            style={{
              color: "var(--autolens-danger)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.82rem",
            }}
          >
            {error}
          </div>
          <Button
            id="try-again-btn"
            size="sm"
            variant="outline"
            onPress={resetPrediction}
            style={{ color: "var(--autolens-danger)", borderColor: "rgba(248,113,113,0.4)" }}
          >
            Try again
          </Button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
