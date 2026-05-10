// ── Header — Logo + active model badge + model selector ──────────────────
// HeroUI v3 Select is a compound component using React Aria primitives.
// We use a native <select> here since HeroUI's Select requires ListBox which
// needs react-aria-components Popover context — simpler and fully functional.
import { Spinner } from "@heroui/react/spinner";
import { useStore } from "../store";

export function Header() {
  const { models, activeModel, isModelSwitching, setActiveModel } = useStore();

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

      {/* Model selector */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", minWidth: "360px" }}>
        {isModelSwitching && <Spinner size="sm" />}
        <div style={{ flex: 1, position: "relative" }}>
          <select
            id="model-selector"
            aria-label="Active model"
            value={activeModel?.id ?? ""}
            onChange={(e) => {
              const id = e.target.value;
              if (id && id !== activeModel?.id) setActiveModel(id);
            }}
            disabled={isModelSwitching || models.length === 0}
            style={{
              width: "100%",
              background: "var(--autolens-surface-2)",
              border: "1px solid var(--autolens-border)",
              borderRadius: "8px",
              padding: "0.45rem 2rem 0.45rem 0.75rem",
              color: "var(--autolens-text)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.8rem",
              cursor: "pointer",
              appearance: "none",
              WebkitAppearance: "none",
              outline: "none",
            }}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.model_name}
                {" — acc "}
                {m.accuracy != null ? (m.accuracy * 100).toFixed(1) + "%" : "—"}
                {" · F1 "}
                {m.f1_macro != null ? (m.f1_macro * 100).toFixed(1) + "%" : "—"}
              </option>
            ))}
          </select>
          {/* Custom chevron */}
          <svg
            style={{
              position: "absolute",
              right: "0.6rem",
              top: "50%",
              transform: "translateY(-50%)",
              pointerEvents: "none",
              color: "var(--autolens-text-muted)",
            }}
            width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5"
          >
            <polyline points="2 4 6 8 10 4"/>
          </svg>
        </div>
      </div>
    </header>
  );
}
