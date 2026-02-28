interface MetricCardProps {
  title: string;
  value: string;
  subtitle: string;
  color: string;
}

export function MetricCard({ title, value, subtitle, color }: MetricCardProps) {
  return (
    <div style={{ ...styles.card, borderTop: `3px solid ${color}` }}>
      <div style={styles.title}>{title}</div>
      <div style={{ ...styles.value, color }}>{value}</div>
      <div style={styles.subtitle}>{subtitle}</div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card: {
    background: "#fff",
    borderRadius: 8,
    padding: "16px 20px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
    minWidth: 180,
  },
  title: { fontSize: 12, color: "#64748b", textTransform: "uppercase" as const, letterSpacing: 0.5 },
  value: { fontSize: 28, fontWeight: 700, margin: "4px 0" },
  subtitle: { fontSize: 13, color: "#94a3b8" },
};
