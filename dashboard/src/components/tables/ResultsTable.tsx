import { useState } from "react";
import type { TestResult } from "../../types";

interface Props {
  results: TestResult[];
  onViewOutput?: (result: TestResult) => void;
}

export function ResultsTable({ results, onViewOutput }: Props) {
  const [sortKey, setSortKey] = useState<keyof TestResult>("model_name");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const toggleSort = (key: keyof TestResult) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const sorted = [...results].sort((a, b) => {
    const av = a[sortKey] ?? "";
    const bv = b[sortKey] ?? "";
    const cmp = av < bv ? -1 : av > bv ? 1 : 0;
    return sortDir === "asc" ? cmp : -cmp;
  });

  const SortIcon = ({ col }: { col: keyof TestResult }) =>
    sortKey === col ? <span>{sortDir === "asc" ? " \u25B2" : " \u25BC"}</span> : null;

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>All Results ({results.length})</h3>
      <div style={styles.tableWrap}>
        <table style={styles.table}>
          <thead>
            <tr>
              {(
                [
                  ["model_name", "Model"],
                  ["prompt_id", "Prompt"],
                  ["category", "Category"],
                  ["ttft_ms", "TTFT"],
                  ["tokens_per_sec", "TPS"],
                  ["output_tokens", "Tokens"],
                  ["cost_usd", "Cost"],
                ] as [keyof TestResult, string][]
              ).map(([key, label]) => (
                <th
                  key={key}
                  style={styles.th}
                  onClick={() => toggleSort(key)}
                >
                  {label}
                  <SortIcon col={key} />
                </th>
              ))}
              <th style={styles.th}>Quality</th>
              <th style={styles.th}>View</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => (
              <tr key={`${r.model_name}-${r.prompt_id}`} style={i % 2 === 0 ? styles.evenRow : undefined}>
                <td style={{ ...styles.td, fontWeight: 500 }}>{r.model_name}</td>
                <td style={styles.td}>{r.prompt_id}</td>
                <td style={styles.td}>{r.category}</td>
                <td style={styles.tdRight}>{r.ttft_ms.toFixed(0)}ms</td>
                <td style={styles.tdRight}>{r.tokens_per_sec.toFixed(1)}</td>
                <td style={styles.tdRight}>{r.output_tokens}</td>
                <td style={styles.tdRight}>{r.cost_usd > 0 ? `$${r.cost_usd.toFixed(4)}` : "free"}</td>
                <td style={{ ...styles.tdRight, color: (r.scores?.weighted ?? 0) >= 7 ? "#22c55e" : "#64748b" }}>
                  {r.scores?.weighted ? `${r.scores.weighted}/10` : "-"}
                  {r.self_bias_flag && " *"}
                </td>
                <td style={styles.tdRight}>
                  <button style={styles.viewBtn} onClick={() => onViewOutput?.(r)}>
                    View
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { background: "#fff", borderRadius: 8, padding: 16, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" },
  title: { fontSize: 14, fontWeight: 600, marginBottom: 12, color: "#1e293b" },
  tableWrap: { overflowX: "auto" },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 12 },
  th: { padding: "8px 10px", textAlign: "left" as const, borderBottom: "2px solid #e2e8f0", color: "#64748b", fontWeight: 600, cursor: "pointer", whiteSpace: "nowrap" as const, userSelect: "none" as const },
  td: { padding: "6px 10px", borderBottom: "1px solid #f1f5f9" },
  tdRight: { padding: "6px 10px", borderBottom: "1px solid #f1f5f9", textAlign: "right" as const },
  evenRow: { background: "#f8fafc" },
  viewBtn: { padding: "2px 8px", border: "1px solid #cbd5e1", borderRadius: 4, background: "#fff", cursor: "pointer", fontSize: 11 },
};
