// ── GradCAMPanel — toggle below SoftmaxChart, only when status==='done' ──
import { AnimatePresence, motion } from "framer-motion";
import { Button } from "@heroui/react/button";
import { Spinner } from "@heroui/react/spinner";
import { useStore } from "../store";

export function GradCAMPanel() {
  const { status, gradcam, isGeneratingGradcam, gradcamVisible, toggleGradcam } = useStore();

  if (status !== "done") return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
      {/* Toggle button */}
      <Button
        id="gradcam-toggle-btn"
        size="sm"
        variant="outline"
        onPress={toggleGradcam}
        isDisabled={isGeneratingGradcam}
        style={{
          alignSelf: "flex-start",
          fontFamily: "var(--font-mono)",
          fontSize: "0.75rem",
          letterSpacing: "0.05em",
        }}
      >
        {isGeneratingGradcam ? (
          <span style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
            <Spinner size="sm" />
            Generating…
          </span>
        ) : gradcamVisible ? (
          "✕ Hide Explanation"
        ) : (
          "🔍 Show Explanation (GradCAM)"
        )}
      </Button>

      {/* Side-by-side overlay image */}
      <AnimatePresence>
        {gradcamVisible && gradcam && (
          <motion.div
            key="gradcam-img"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.35 }}
            style={{
              background: "var(--autolens-surface)",
              border: "1px solid var(--autolens-border)",
              borderRadius: "10px",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                padding: "0.6rem 1rem",
                fontSize: "0.62rem",
                color: "var(--autolens-text-muted)",
                letterSpacing: "0.12em",
                borderBottom: "1px solid var(--autolens-border)",
              }}
            >
              ORIGINAL &nbsp;|&nbsp; GRADCAM OVERLAY
            </div>
            <img
              src={gradcam}
              alt="GradCAM: original and heatmap overlay side by side"
              style={{ width: "100%", display: "block" }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
