// ── App — 3-column layout, mount fetchModels, ErrorBoundary wrapper ───────
import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Header } from "./components/Header";
import { ImageUpload } from "./components/ImageUpload";
import { ResultCard } from "./components/ResultCard";
import { SoftmaxChart } from "./components/SoftmaxChart";
import { GradCAMPanel } from "./components/GradCAMPanel";
import { ModelDetailPanel } from "./components/ModelDetailPanel";
import { useStore } from "./store";

function Layout() {
  const { fetchModels } = useStore();

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <Header />

      {/* 3-column grid — centered */}
      <main
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "2rem 1.5rem 2.5rem",
        }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "320px 1fr 300px",
            gap: "1.25rem",
            maxWidth: "1100px",
            margin: "0 auto",
            alignItems: "start",
          }}
        >
        {/* LEFT — upload */}
        <AnimatePresence>
          <motion.div
            key="left-col"
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.05 }}
            style={{ display: "flex", flexDirection: "column", gap: "1rem" }}
          >
            <SectionLabel>VEHICLE IMAGE</SectionLabel>
            <ImageUpload />
          </motion.div>
        </AnimatePresence>

        {/* CENTRE — results */}
        <motion.div
          key="centre-col"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          style={{ display: "flex", flexDirection: "column", gap: "1rem" }}
        >
          <SectionLabel>PREDICTION</SectionLabel>
          <ResultCard />
          <SoftmaxChart />
          <GradCAMPanel />
        </motion.div>

        {/* RIGHT — model details */}
        <motion.div
          key="right-col"
          initial={{ opacity: 0, x: 12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.15 }}
        >
          <SectionLabel>MODEL INFO</SectionLabel>
          <div style={{ marginTop: "0.5rem" }}>
            <ModelDetailPanel />
          </div>
        </motion.div>
        </div>  {/* end centered grid */}
      </main>
    </div>
  );
}

function SectionLabel({ children }: { children: string }) {
  return (
    <div
      style={{
        fontSize: "0.6rem",
        letterSpacing: "0.14em",
        color: "var(--autolens-text-muted)",
        fontFamily: "var(--font-mono)",
        paddingBottom: "0.25rem",
        borderBottom: "1px solid var(--autolens-border)",
      }}
    >
      {children}
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
        {/* Mobile guard */}
        <div className="mobile-notice">
          <span style={{ fontSize: "2rem" }}>🖥️</span>
          <span style={{ fontFamily: "var(--font-display)", fontSize: "1rem" }}>
            AutoLens AI is designed for desktop use.
          </span>
          <span style={{ fontSize: "0.8rem" }}>Please open on a screen wider than 1024px.</span>
        </div>

        {/* Desktop app */}
        <div className="desktop-only" style={{ height: "100%" }}>
          <Layout />
        </div>
      </ErrorBoundary>
  );
}
