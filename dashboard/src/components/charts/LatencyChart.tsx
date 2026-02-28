import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { TestResult } from "../../types";

interface Props {
  results: TestResult[];
}

export function LatencyChart({ results }: Props) {
  const byModel = new Map<string, { sum: number; count: number }>();
  results.forEach((r) => {
    const e = byModel.get(r.model_name) || { sum: 0, count: 0 };
    e.sum += r.ttft_ms;
    e.count += 1;
    byModel.set(r.model_name, e);
  });

  const data = [...byModel.entries()]
    .map(([name, v]) => ({ name, ttft: +(v.sum / v.count).toFixed(0) }))
    .sort((a, b) => a.ttft - b.ttft);

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Time to First Token (ms)</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} layout="vertical" margin={{ left: 120, right: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis type="number" tick={{ fontSize: 12 }} />
          <YAxis dataKey="name" type="category" tick={{ fontSize: 12 }} width={110} />
          <Tooltip formatter={(v) => [`${v}ms`, "TTFT"]} />
          <Bar dataKey="ttft" radius={[0, 4, 4, 0]}>
            {data.map((d) => (
              <Cell
                key={d.name}
                fill={d.name.includes("Claude") ? "#8b5cf6" : "#f59e0b"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { background: "#fff", borderRadius: 8, padding: 16, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" },
  title: { fontSize: 14, fontWeight: 600, marginBottom: 8, color: "#1e293b" },
};
