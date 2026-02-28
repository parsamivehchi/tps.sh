import { useMemo } from "react";
import type { TestResult } from "../../types";

interface Props {
  results: TestResult[];
}

interface ModelSummary {
  name: string;
  avgTtft: number;
  avgTps: number;
  avgQuality: number;
  totalCost: number;
  testCount: number;
}

export function RankingTable({ results }: Props) {
  const summaries = useMemo(() => {
    const byModel = new Map<string, TestResult[]>();
    results.forEach((r) => {
      const arr = byModel.get(r.model_name) || [];
      arr.push(r);
      byModel.set(r.model_name, arr);
    });

    const rows: ModelSummary[] = [...byModel.entries()].map(([name, rs]) => ({
      name,
      avgTtft: +(rs.reduce((s, r) => s + r.ttft_ms, 0) / rs.length).toFixed(0),
      avgTps: +(rs.reduce((s, r) => s + r.tokens_per_sec, 0) / rs.length).toFixed(1),
      avgQuality: +(
        rs.filter((r) => r.scores?.weighted).reduce((s, r) => s + (r.scores?.weighted ?? 0), 0) /
        (rs.filter((r) => r.scores?.weighted).length || 1)
      ).toFixed(2),
      totalCost: +rs.reduce((s, r) => s + r.cost_usd, 0).toFixed(4),
      testCount: rs.length,
    }));

    return rows.sort((a, b) => b.avgQuality - a.avgQuality || b.avgTps - a.avgTps);
  }, [results]);

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Model Rankings</h3>
      <table style={styles.table}>
        <thead>
          <tr>
            <th style={styles.th}>#</th>
            <th style={{ ...styles.th, textAlign: "left" }}>Model</th>
            <th style={styles.th}>Avg TTFT</th>
            <th style={styles.th}>Avg TPS</th>
            <th style={styles.th}>Quality</th>
            <th style={styles.th}>Total Cost</th>
            <th style={styles.th}>Tests</th>
          </tr>
        </thead>
        <tbody>
          {summaries.map((s, i) => (
            <tr key={s.name} style={i % 2 === 0 ? styles.evenRow : undefined}>
              <td style={styles.td}>{i + 1}</td>
              <td style={{ ...styles.td, textAlign: "left", fontWeight: 600 }}>{s.name}</td>
              <td style={styles.td}>{s.avgTtft}ms</td>
              <td style={styles.td}>{s.avgTps}</td>
              <td style={{ ...styles.td, color: s.avgQuality >= 7 ? "#22c55e" : s.avgQuality >= 5 ? "#f59e0b" : "#64748b" }}>
                {s.avgQuality > 0 ? `${s.avgQuality}/10` : "-"}
              </td>
              <td style={styles.td}>{s.totalCost > 0 ? `$${s.totalCost}` : "free"}</td>
              <td style={styles.td}>{s.testCount}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { background: "#fff", borderRadius: 8, padding: 16, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" },
  title: { fontSize: 14, fontWeight: 600, marginBottom: 12, color: "#1e293b" },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 13 },
  th: { padding: "8px 12px", textAlign: "right" as const, borderBottom: "2px solid #e2e8f0", color: "#64748b", fontWeight: 600, fontSize: 12 },
  td: { padding: "8px 12px", textAlign: "right" as const, borderBottom: "1px solid #f1f5f9" },
  evenRow: { background: "#f8fafc" },
};
