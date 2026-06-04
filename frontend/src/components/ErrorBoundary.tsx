// ── ErrorBoundary ─────────────────────────────────────────────────────────
import { Component, type ReactNode } from "react";

interface Props { children: ReactNode }
interface State { hasError: boolean; message: string }

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(err: Error): State {
    return { hasError: true, message: err.message || "Unknown render error" };
  }

  override render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            height: "100vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "1rem",
            background: "var(--autolens-bg)",
            color: "var(--autolens-danger)",
            fontFamily: "var(--font-mono)",
            padding: "2rem",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: "1.2rem" }}>Render error</div>
          <div style={{ fontSize: "0.8rem", color: "var(--autolens-text-muted)", maxWidth: "500px" }}>
            {this.state.message}
          </div>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: "0.5rem",
              padding: "0.4rem 1rem",
              background: "transparent",
              border: "1px solid rgba(248,113,113,0.4)",
              borderRadius: "6px",
              color: "var(--autolens-danger)",
              cursor: "pointer",
              fontFamily: "var(--font-mono)",
            }}
          >
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
