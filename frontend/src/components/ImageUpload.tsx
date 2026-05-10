// ── ImageUpload — Drag & drop + click, preview thumbnail ─────────────────
import { useCallback, useRef, useState } from "react";
import { Button } from "@heroui/react/button";
import { useStore } from "../store";

export function ImageUpload() {
  const { selectedFile, setFile, predict, status, isModelSwitching } = useStore();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const preview = selectedFile ? URL.createObjectURL(selectedFile) : null;
  const isAnalyzing = status === "analyzing";
  const canClassify = !!selectedFile && !isAnalyzing && !isModelSwitching;

  const onFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      setFile(files[0]);
    },
    [setFile]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      onFiles(e.dataTransfer.files);
    },
    [onFiles]
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
      {/* Drop zone */}
      <div
        id="image-dropzone"
        role="button"
        tabIndex={0}
        aria-label="Upload vehicle image"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        style={{
          position: "relative",
          width: "100%",
          aspectRatio: "4/3",
          background: isDragging ? "rgba(99,102,241,0.08)" : "var(--autolens-surface-2)",
          border: `1.5px dashed ${isDragging ? "var(--autolens-accent)" : "var(--autolens-border)"}`,
          borderRadius: "10px",
          cursor: "pointer",
          overflow: "hidden",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transition: "border-color 0.2s, background 0.2s",
        }}
      >
        {preview ? (
          <>
            <img
              src={preview}
              alt="Selected vehicle"
              style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
            />
            {/* × clear button */}
            <button
              id="clear-image-btn"
              onClick={(e) => { e.stopPropagation(); setFile(null); }}
              aria-label="Clear image"
              style={{
                position: "absolute",
                top: "8px",
                right: "8px",
                width: "26px",
                height: "26px",
                borderRadius: "50%",
                background: "rgba(13,15,18,0.75)",
                border: "1px solid var(--autolens-border)",
                color: "var(--autolens-text-dim)",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "0.85rem",
                lineHeight: 1,
                backdropFilter: "blur(4px)",
                transition: "background 0.15s, color 0.15s",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLButtonElement).style.background = "rgba(212,105,106,0.25)";
                (e.currentTarget as HTMLButtonElement).style.color = "var(--autolens-danger)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLButtonElement).style.background = "rgba(13,15,18,0.75)";
                (e.currentTarget as HTMLButtonElement).style.color = "var(--autolens-text-dim)";
              }}
            >
              ×
            </button>
          </>
        ) : (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "0.5rem",
              color: "var(--autolens-text-muted)",
              userSelect: "none",
            }}
          >
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
            </svg>
            <span style={{ fontSize: "0.8rem" }}>Drop image here or click to upload</span>
            <span style={{ fontSize: "0.7rem", color: "var(--autolens-text-muted)" }}>
              JPEG · PNG · WebP · max 10 MB
            </span>
          </div>
        )}

        {/* Analyzing overlay */}
        {isAnalyzing && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              background: "rgba(8,8,16,0.65)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: "0.5rem",
              backdropFilter: "blur(2px)",
            }}
          >
            <div className="scanning-line" />
            <span
              className="pulse-dim"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.8rem",
                color: "var(--autolens-accent)",
                letterSpacing: "0.1em",
              }}
            >
              ANALYZING…
            </span>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        id="file-input"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        style={{ display: "none" }}
        onChange={(e) => onFiles(e.target.files)}
      />

      {/* Classify button */}
      <Button
        id="classify-btn"
        onPress={predict}
        isDisabled={!canClassify}
        variant="primary"
        size="md"
        style={{ width: "100%", fontFamily: "var(--font-mono)", letterSpacing: "0.08em" }}
      >
        {isModelSwitching ? "Switching model…" : isAnalyzing ? "Analyzing…" : "Classify"}
      </Button>
    </div>
  );
}
