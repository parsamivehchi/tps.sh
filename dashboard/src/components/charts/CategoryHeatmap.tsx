import type { TestResult } from "../../types";

interface Props {
  results: TestResult[];
  categories: Record<string, string>;
}

export function CategoryHeatmap({ results, categories }: Props) {
  const scored = results.filter((r) => r.scores?.weighted);
  if (scored.length === 0) return null;

  const models = [...new Set(scored.map((r) => r.model_name))];
  const cats = Object.keys(categories);

  // Build heatmap: model x category -> avg score
  const grid: Record<string, Record<string, number>> = {};
  cats.forEach((cat) => {
    grid[cat] = {};
    models.forEach((m) => {
      const mr = scored.filter((r) => r.model_name === m && r.category === cat);
      grid[cat][m] = mr.length > 0
        ? +(mr.reduce((s, r) => s + (r.scores?.weighted ?? 0), 0) / mr.length).toFixed(1)
        : 0;
    });
  });

  const getColor = (score: number) => {
    if (score === 0) return "#f1f5f9";
    if (score >= 8) return "#22c55e";
    if (score >= 6) return "#84cc16";
    if (score >= 4) return "#f59e0b";
    return "#ef4444";
  };

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Quality Heatmap (Category x Model)</h3>
      <div style={styles.grid}>
        <div style={styles.headerRow}>
          <div style={styles.cornerCell} />
          {models.map((m) => (
            <div key={m} style={styles.headerCell}>{m}</div>
          ))}
        </div>
        {cats.map((cat) => (
          <div key={cat} style={styles.row}>
            <div style={styles.rowLabel}>{categories[cat]}</div>
            {models.map((m) => {
              const score = grid[cat][m];
              return (
                <div
                  key={m}
                  style={{
                    ...styles.cell,
                    background: getColor(score),
                    color: score >= 6 ? "#fff" : "#1e293b",
                  }}
                  title={`${m}: ${score}/10`}
                >
                  {score > 0 ? score : "-"}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { background: "#fff", borderRadius: 8, padding: 16, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" },
  title: { fontSize: 14, fontWeight: 600, marginBottom: 12, color: "#1e293b" },
  grid: { overflowX: "auto" },
  headerRow: { display: "flex", gap: 2, marginBottom: 2 },
  cornerCell: { width: 140, flexShrink: 0 },
  headerCell: { flex: 1, minWidth: 100, fontSize: 11, fontWeight: 600, textAlign: "center" as const, padding: "6px 2px", color: "#475569" },
  row: { display: "flex", gap: 2, marginBottom: 2 },
  rowLabel: { width: 140, flexShrink: 0, fontSize: 12, padding: "8px 4px", color: "#475569" },
  cell: { flex: 1, minWidth: 100, textAlign: "center" as const, padding: "8px 4px", borderRadius: 4, fontSize: 13, fontWeight: 600 },
};
