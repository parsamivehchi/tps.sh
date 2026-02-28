import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ZAxis,
} from "recharts";
import type { TestResult } from "../../types";

interface Props {
  results: TestResult[];
}

const COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"];

export function CostEfficiency({ results }: Props) {
  const scored = results.filter((r) => r.scores?.weighted);
  if (scored.length === 0) return null;

  const models = [...new Set(scored.map((r) => r.model_name))];
  const modelData = models.map((m) => {
    const mr = scored.filter((r) => r.model_name === m);
    const avgQ = mr.reduce((s, r) => s + (r.scores?.weighted ?? 0), 0) / mr.length;
    const totalCost = mr.reduce((s, r) => s + r.cost_usd, 0);
    const avgTps = mr.reduce((s, r) => s + r.tokens_per_sec, 0) / mr.length;
    return { name: m, quality: +avgQ.toFixed(2), cost: +totalCost.toFixed(4), tps: +avgTps.toFixed(1) };
  });

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Cost vs Quality</h3>
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ left: 10, right: 20, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="cost" name="Cost ($)" tick={{ fontSize: 12 }} label={{ value: "Cost ($)", position: "bottom", fontSize: 12 }} />
          <YAxis dataKey="quality" name="Quality" tick={{ fontSize: 12 }} domain={[0, 10]} label={{ value: "Quality", angle: -90, position: "insideLeft", fontSize: 12 }} />
          <ZAxis dataKey="tps" range={[60, 300]} name="TPS" />
          <Tooltip
            formatter={(value, name) => {
              if (name === "Cost ($)") return [`$${value}`, name];
              if (name === "Quality") return [`${value}/10`, name];
              return [`${value} tok/s`, name];
            }}
            labelFormatter={(_: unknown, payload: ReadonlyArray<{payload?: {name?: string}}>) => payload?.[0]?.payload?.name ?? ""}
          />
          <Scatter data={modelData}>
            {modelData.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <div style={styles.legend}>
        {modelData.map((d, i) => (
          <span key={d.name} style={styles.legendItem}>
            <span style={{ ...styles.dot, background: COLORS[i % COLORS.length] }} />
            {d.name}
          </span>
        ))}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { background: "#fff", borderRadius: 8, padding: 16, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" },
  title: { fontSize: 14, fontWeight: 600, marginBottom: 8, color: "#1e293b" },
  legend: { display: "flex", flexWrap: "wrap", gap: 12, marginTop: 8, justifyContent: "center" },
  legendItem: { display: "flex", alignItems: "center", gap: 4, fontSize: 12 },
  dot: { width: 8, height: 8, borderRadius: "50%", display: "inline-block" },
};
