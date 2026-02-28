import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";
import type { TestResult } from "../../types";

interface Props {
  results: TestResult[];
}

const MODEL_COLORS = [
  "#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4",
];

export function QualityRadar({ results }: Props) {
  const scored = results.filter((r) => r.scores?.weighted);
  if (scored.length === 0) return null;

  // Group by model, compute avg scores per dimension
  const models = [...new Set(scored.map((r) => r.model_name))];
  const dims = ["correctness", "completeness", "clarity"];

  const radarData = dims.map((dim) => {
    const entry: Record<string, string | number> = { dimension: dim.charAt(0).toUpperCase() + dim.slice(1) };
    models.forEach((m) => {
      const modelResults = scored.filter((r) => r.model_name === m);
      const avg =
        modelResults.reduce(
          (s, r) => s + ((r.scores as unknown as Record<string, number>)?.[dim] ?? 0),
          0
        ) / modelResults.length;
      entry[m] = +avg.toFixed(2);
    });
    return entry;
  });

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Quality Radar</h3>
      <ResponsiveContainer width="100%" height={320}>
        <RadarChart data={radarData}>
          <PolarGrid stroke="#e2e8f0" />
          <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 12 }} />
          <PolarRadiusAxis domain={[0, 10]} tick={{ fontSize: 10 }} />
          <Tooltip />
          {models.map((m, i) => (
            <Radar
              key={m}
              name={m}
              dataKey={m}
              stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
              fill={MODEL_COLORS[i % MODEL_COLORS.length]}
              fillOpacity={0.15}
            />
          ))}
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { background: "#fff", borderRadius: 8, padding: 16, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" },
  title: { fontSize: 14, fontWeight: 600, marginBottom: 8, color: "#1e293b" },
};
