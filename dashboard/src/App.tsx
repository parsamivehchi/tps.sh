import { useData } from "./hooks/useData";
import { useFilters } from "./hooks/useFilters";
import { Sidebar } from "./components/Sidebar";
import { Dashboard } from "./components/Dashboard";

export default function App() {
  const { data, loading, error } = useData();
  const {
    filters,
    allModels,
    allCategories,
    toggleModel,
    toggleCategory,
    resetFilters,
    filteredResults,
  } = useFilters(data);

  if (loading) {
    return (
      <div style={styles.center}>
        <div style={styles.spinner} />
        <p style={styles.loadText}>Loading benchmark data...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div style={styles.center}>
        <h2 style={{ color: "#ef4444" }}>Failed to load data</h2>
        <p style={{ color: "#64748b" }}>
          {error || "No data found."} Make sure{" "}
          <code>public/data/dashboard_data.json</code> exists.
        </p>
        <p style={{ color: "#94a3b8", fontSize: 13 }}>
          Run: <code>python -m llm_bench export &lt;run_id&gt;</code>
        </p>
      </div>
    );
  }

  return (
    <div style={styles.layout}>
      <Sidebar
        data={data}
        filters={filters}
        allModels={allModels}
        allCategories={allCategories}
        toggleModel={toggleModel}
        toggleCategory={toggleCategory}
        resetFilters={resetFilters}
      />
      <Dashboard data={data} filteredResults={filteredResults} />
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  layout: {
    display: "flex",
    height: "100vh",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  center: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    gap: 8,
  },
  spinner: {
    width: 32,
    height: 32,
    border: "3px solid #e2e8f0",
    borderTopColor: "#3b82f6",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
  loadText: { color: "#64748b", fontSize: 14 },
};
