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

const COLORS: Record<string, string> = {
  ollama: "#22c55e",
  anthropic: "#3b82f6",
};

export function SpeedChart({ results }: Props) {
  const byModel = new Map<string, { sum: number; count: number; provider: string }>();
  results.forEach((r) => {
    const entry = byModel.get(r.model_name) || { sum: 0, count: 0, provider: "ollama" };
    entry.sum += r.tokens_per_sec;
    entry.count += 1;
    byModel.set(r.model_name, entry);
  });

  const data = [...byModel.entries()]
    .map(([name, v]) => ({
      name,
      tps: +(v.sum / v.count).toFixed(1),
      provider: v.provider,
    }))
    .sort((a, b) => b.tps - a.tps);

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Generation Speed (tokens/sec)</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} layout="vertical" margin={{ left: 120, right: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis type="number" tick={{ fontSize: 12 }} />
          <YAxis dataKey="name" type="category" tick={{ fontSize: 12 }} width={110} />
          <Tooltip formatter={(v) => [`${v} tok/s`, "Speed"]} />
          <Bar dataKey="tps" radius={[0, 4, 4, 0]}>
            {data.map((d) => (
              <Cell
                key={d.name}
                fill={d.name.includes("Claude") ? COLORS.anthropic : COLORS.ollama}
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
