import type { TestResult } from "../../types";

interface Props {
  result: TestResult;
  onClose: () => void;
}

export function OutputViewer({ result, onClose }: Props) {
  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <div>
            <h3 style={styles.title}>
              {result.model_name} — {result.prompt_id}
            </h3>
            <div style={styles.meta}>
              Category: {result.category} | TTFT: {result.ttft_ms.toFixed(0)}ms |
              TPS: {result.tokens_per_sec.toFixed(1)} | Tokens: {result.output_tokens}
              {result.cost_usd > 0 && ` | Cost: $${result.cost_usd.toFixed(4)}`}
            </div>
          </div>
          <button style={styles.closeBtn} onClick={onClose}>
            \u2715
          </button>
        </div>

        {result.scores && (
          <div style={styles.scores}>
            <div style={styles.scoreItem}>
              <span style={styles.scoreLabel}>Correctness</span>
              <span style={styles.scoreValue}>{result.scores.correctness}/10</span>
            </div>
            <div style={styles.scoreItem}>
              <span style={styles.scoreLabel}>Completeness</span>
              <span style={styles.scoreValue}>{result.scores.completeness}/10</span>
            </div>
            <div style={styles.scoreItem}>
              <span style={styles.scoreLabel}>Clarity</span>
              <span style={styles.scoreValue}>{result.scores.clarity}/10</span>
            </div>
            <div style={styles.scoreItem}>
              <span style={styles.scoreLabel}>Weighted</span>
              <span style={{ ...styles.scoreValue, fontWeight: 700, fontSize: 16 }}>
                {result.scores.weighted}/10
              </span>
            </div>
            {result.self_bias_flag && (
              <div style={styles.biasTag}>Claude-judging-Claude</div>
            )}
          </div>
        )}

        {result.scores?.reasoning && (
          <div style={styles.reasoning}>
            <strong>Judge Reasoning:</strong> {result.scores.reasoning}
          </div>
        )}

        <div style={styles.outputWrap}>
          <pre style={styles.output}>{result.output}</pre>
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)",
    display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000,
  },
  modal: {
    background: "#fff", borderRadius: 12, width: "90vw", maxWidth: 900,
    maxHeight: "85vh", display: "flex", flexDirection: "column", overflow: "hidden",
  },
  header: {
    display: "flex", justifyContent: "space-between", alignItems: "start",
    padding: "16px 20px", borderBottom: "1px solid #e2e8f0",
  },
  title: { margin: 0, fontSize: 16, color: "#1e293b" },
  meta: { fontSize: 12, color: "#64748b", marginTop: 4 },
  closeBtn: {
    background: "none", border: "none", fontSize: 20, cursor: "pointer",
    color: "#94a3b8", padding: "0 4px",
  },
  scores: {
    display: "flex", gap: 16, padding: "12px 20px", background: "#f8fafc",
    borderBottom: "1px solid #e2e8f0", flexWrap: "wrap", alignItems: "center",
  },
  scoreItem: { display: "flex", flexDirection: "column", alignItems: "center", gap: 2 },
  scoreLabel: { fontSize: 10, color: "#64748b", textTransform: "uppercase" as const },
  scoreValue: { fontSize: 14, fontWeight: 600, color: "#1e293b" },
  biasTag: {
    fontSize: 11, padding: "2px 8px", borderRadius: 4,
    background: "#fef3c7", color: "#92400e",
  },
  reasoning: { padding: "12px 20px", fontSize: 13, color: "#475569", background: "#fffbeb", borderBottom: "1px solid #e2e8f0" },
  outputWrap: { flex: 1, overflow: "auto", padding: "12px 20px" },
  output: {
    fontSize: 12, lineHeight: 1.5, whiteSpace: "pre-wrap" as const,
    wordBreak: "break-word" as const, margin: 0, fontFamily: "'SF Mono', 'Fira Code', monospace",
  },
};
