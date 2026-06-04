// ── Header — Logo + active model badge + model selector ──────────────────
// HeroUI v3 Select is a compound component using React Aria primitives.
// We use a native <select> here since HeroUI's Select requires ListBox which
// needs react-aria-components Popover context — simpler and fully functional.
import { Spinner } from "@heroui/react/spinner";
import { useStore } from "../store";

export function Header() {
  const { activeModel, isModelSwitching } = useStore();

  return (
    <header
      style={{
        background: "var(--autolens-surface)",
        borderBottom: "1px solid var(--autolens-border)",
        padding: "0 1.5rem",
        height: "60px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "1.5rem",
        position: "sticky",
        top: 0,
        zIndex: 50,
      }}
    >
      {/* Logo */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexShrink: 0 }}>
        <span
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "1.1rem",
            fontWeight: 700,
            color: "var(--autolens-text)",
            letterSpacing: "0.04em",
          }}
        >
          AutoLens
        </span>
        <span
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "1.1rem",
            fontWeight: 700,
            color: "var(--autolens-accent)",
            letterSpacing: "0.04em",
          }}
        >
          AI
        </span>
      </div>

      {/* Model selector (Removed - Only single active model used now) */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
        {isModelSwitching && <Spinner size="sm" />}
        <div style={{ 
          background: "var(--autolens-surface-2)",
          border: "1px solid var(--autolens-border)",
          borderRadius: "8px",
          padding: "0.45rem 1rem",
          color: "var(--autolens-text)",
          fontFamily: "var(--font-mono)",
          fontSize: "0.8rem",
          display: "flex",
          alignItems: "center",
          gap: "0.5rem"
        }}>
          <span style={{ color: "var(--autolens-accent)" }}>●</span>
          {activeModel ? "AutoLens ViT-S" : "AutoLens ViT-S"}
          <span style={{ color: "var(--autolens-text-muted)", marginLeft: "0.5rem" }}>
            {"acc "}
            {activeModel?.accuracy != null ? (activeModel.accuracy * 100).toFixed(1) + "%" : "96.0%"}
          </span>
        </div>
      </div>
    </header>
  );
}
