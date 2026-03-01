import { useState } from "react";
import type { DashboardData, TestResult } from "../types";
import { MetricCard } from "./cards/MetricCard";
import { SpeedChart } from "./charts/SpeedChart";
import { LatencyChart } from "./charts/LatencyChart";
import { QualityRadar } from "./charts/QualityRadar";
import { CostEfficiency } from "./charts/CostEfficiency";
import { CategoryHeatmap } from "./charts/CategoryHeatmap";
import { RankingTable } from "./tables/RankingTable";
import { ResultsTable } from "./tables/ResultsTable";
import { OutputViewer } from "./detail/OutputViewer";

interface Props {
  data: DashboardData;
  filteredResults: TestResult[];
}

export function Dashboard({ data, filteredResults }: Props) {
  const [selectedResult, setSelectedResult] = useState<TestResult | null>(null);
  const results = filteredResults;

  // Compute highlight metrics
  const models = [...new Set(results.map((r) => r.model_name))];
  const modelStats = models.map((m) => {
    const mr = results.filter((r) => r.model_name === m);
    return {
      name: m,
      avgTps: mr.reduce((s, r) => s + r.tokens_per_sec, 0) / mr.length,
      avgTtft: mr.reduce((s, r) => s + r.ttft_ms, 0) / mr.length,
      avgQuality:
        mr.filter((r) => r.scores?.weighted).length > 0
          ? mr
              .filter((r) => r.scores?.weighted)
              .reduce((s, r) => s + (r.scores?.weighted ?? 0), 0) /
            mr.filter((r) => r.scores?.weighted).length
          : 0,
      totalCost: mr.reduce((s, r) => s + r.cost_usd, 0),
    };
  });

  const fastest = modelStats.length
    ? modelStats.reduce((a, b) => (a.avgTtft < b.avgTtft ? a : b))
    : null;
  const highestTps = modelStats.length
    ? modelStats.reduce((a, b) => (a.avgTps > b.avgTps ? a : b))
    : null;
  const bestQuality = modelStats.filter((m) => m.avgQuality > 0).length
    ? modelStats
        .filter((m) => m.avgQuality > 0)
        .reduce((a, b) => (a.avgQuality > b.avgQuality ? a : b))
    : null;
  const bestValue = modelStats.length
    ? modelStats.reduce((a, b) => {
        const av = a.totalCost > 0 ? a.avgQuality / a.totalCost : a.avgQuality * a.avgTps;
        const bv = b.totalCost > 0 ? b.avgQuality / b.totalCost : b.avgQuality * b.avgTps;
        return av > bv ? a : b;
      })
    : null;

  return (
    <div style={styles.main}>
      {/* Header */}
      <div style={styles.header}>
        <div>
          <h1 style={styles.h1}>tps.sh Dashboard</h1>
          <p style={styles.subtitle}>
            Run: {data.meta.run_id} | {data.meta.total_tests} tests |{" "}
            {data.meta.model_count} models
          </p>
        </div>
      </div>

      {/* Metric Cards */}
      <div style={styles.cards}>
        {fastest && (
          <MetricCard
            title="Fastest TTFT"
            value={`${fastest.avgTtft.toFixed(0)}ms`}
            subtitle={fastest.name}
            color="#8b5cf6"
          />
        )}
        {highestTps && (
          <MetricCard
            title="Highest TPS"
            value={`${highestTps.avgTps.toFixed(1)}`}
            subtitle={highestTps.name}
            color="#22c55e"
          />
        )}
        {bestQuality && (
          <MetricCard
            title="Best Quality"
            value={`${bestQuality.avgQuality.toFixed(1)}/10`}
            subtitle={bestQuality.name}
            color="#3b82f6"
          />
        )}
        {bestValue && (
          <MetricCard
            title="Best Value"
            value={bestValue.totalCost > 0 ? `$${bestValue.totalCost.toFixed(4)}` : "free"}
            subtitle={bestValue.name}
            color="#f59e0b"
          />
        )}
      </div>

      {/* Charts Row 1 */}
      <div style={styles.chartRow}>
        <div style={styles.chartHalf}>
          <SpeedChart results={results} />
        </div>
        <div style={styles.chartHalf}>
          <LatencyChart results={results} />
        </div>
      </div>

      {/* Charts Row 2 */}
      <div style={styles.chartRow}>
        <div style={styles.chartHalf}>
          <QualityRadar results={results} />
        </div>
        <div style={styles.chartHalf}>
          <CostEfficiency results={results} />
        </div>
      </div>

      {/* Heatmap */}
      <CategoryHeatmap results={results} categories={data.categories} />

      {/* Rankings */}
      <RankingTable results={results} />

      {/* Full Results */}
      <ResultsTable results={results} onViewOutput={setSelectedResult} />

      {/* Bias Disclaimer */}
      <div style={styles.disclaimer}>
        * Quality scores from Claude models are flagged — Claude Sonnet 4.6
        judges all outputs including its own family. Interpret Claude scores with
        this caveat.
      </div>

      {/* Output Viewer Modal */}
      {selectedResult && (
        <OutputViewer result={selectedResult} onClose={() => setSelectedResult(null)} />
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  main: {
    flex: 1,
    padding: 24,
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: 20,
    background: "#f1f5f9",
  },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center" },
  h1: { margin: 0, fontSize: 22, fontWeight: 700, color: "#0f172a" },
  subtitle: { margin: "4px 0 0", fontSize: 13, color: "#64748b" },
  cards: { display: "flex", gap: 16, flexWrap: "wrap" },
  chartRow: { display: "flex", gap: 16 },
  chartHalf: { flex: 1, minWidth: 0 },
  disclaimer: {
    fontSize: 12,
    color: "#94a3b8",
    textAlign: "center",
    padding: "12px 0",
    borderTop: "1px solid #e2e8f0",
  },
};
