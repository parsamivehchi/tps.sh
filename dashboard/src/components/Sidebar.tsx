import type { DashboardData, FilterState } from "../types";

interface SidebarProps {
  data: DashboardData;
  filters: FilterState;
  allModels: string[];
  allCategories: string[];
  toggleModel: (m: string) => void;
  toggleCategory: (c: string) => void;
  resetFilters: () => void;
}

export function Sidebar({
  data,
  filters,
  allModels,
  allCategories,
  toggleModel,
  toggleCategory,
  resetFilters,
}: SidebarProps) {
  return (
    <aside style={styles.sidebar}>
      <h3 style={styles.heading}>Filters</h3>

      <div style={styles.section}>
        <h4 style={styles.subheading}>Models</h4>
        {allModels.map((m) => {
          const info = data.models[m];
          const active =
            filters.models.length === 0 || filters.models.includes(m);
          return (
            <label key={m} style={{ ...styles.label, opacity: active ? 1 : 0.4 }}>
              <input
                type="checkbox"
                checked={active}
                onChange={() => toggleModel(m)}
                style={styles.checkbox}
              />
              <span>
                {m}
                <span style={styles.badge}>
                  {info?.provider === "ollama" ? "local" : "cloud"}
                </span>
              </span>
            </label>
          );
        })}
      </div>

      <div style={styles.section}>
        <h4 style={styles.subheading}>Categories</h4>
        {allCategories.map((c) => {
          const active =
            filters.categories.length === 0 || filters.categories.includes(c);
          return (
            <label key={c} style={{ ...styles.label, opacity: active ? 1 : 0.4 }}>
              <input
                type="checkbox"
                checked={active}
                onChange={() => toggleCategory(c)}
                style={styles.checkbox}
              />
              {data.categories[c]}
            </label>
          );
        })}
      </div>

      <button onClick={resetFilters} style={styles.reset}>
        Reset All
      </button>
    </aside>
  );
}

const styles: Record<string, React.CSSProperties> = {
  sidebar: {
    width: 240,
    padding: "16px",
    borderRight: "1px solid #e2e8f0",
    background: "#f8fafc",
    overflowY: "auto",
    flexShrink: 0,
  },
  heading: { margin: "0 0 16px", fontSize: 16, color: "#1e293b" },
  subheading: { margin: "0 0 8px", fontSize: 13, color: "#64748b", textTransform: "uppercase" as const, letterSpacing: 1 },
  section: { marginBottom: 20 },
  label: { display: "flex", alignItems: "center", gap: 6, marginBottom: 6, fontSize: 13, cursor: "pointer" },
  checkbox: { accentColor: "#3b82f6" },
  badge: {
    fontSize: 10,
    padding: "1px 5px",
    borderRadius: 4,
    background: "#e2e8f0",
    color: "#475569",
    marginLeft: 4,
  },
  reset: {
    width: "100%",
    padding: "8px",
    border: "1px solid #cbd5e1",
    borderRadius: 6,
    background: "#fff",
    cursor: "pointer",
    fontSize: 13,
  },
};
