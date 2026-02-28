import { useState, useMemo } from "react";
import type { DashboardData, FilterState, TestResult } from "../types";

export function useFilters(data: DashboardData | null) {
  const allModels = useMemo(
    () => (data ? Object.keys(data.models) : []),
    [data]
  );
  const allCategories = useMemo(
    () => (data ? Object.keys(data.categories) : []),
    [data]
  );

  const [filters, setFilters] = useState<FilterState>({
    models: [],
    categories: [],
  });

  const toggleModel = (model: string) => {
    setFilters((prev) => ({
      ...prev,
      models: prev.models.includes(model)
        ? prev.models.filter((m) => m !== model)
        : [...prev.models, model],
    }));
  };

  const toggleCategory = (cat: string) => {
    setFilters((prev) => ({
      ...prev,
      categories: prev.categories.includes(cat)
        ? prev.categories.filter((c) => c !== cat)
        : [...prev.categories, cat],
    }));
  };

  const resetFilters = () => setFilters({ models: [], categories: [] });

  const filteredResults: TestResult[] = useMemo(() => {
    if (!data) return [];
    return data.results.filter((r) => {
      const modelMatch =
        filters.models.length === 0 || filters.models.includes(r.model_name);
      const catMatch =
        filters.categories.length === 0 ||
        filters.categories.includes(r.category);
      return modelMatch && catMatch && !r.error;
    });
  }, [data, filters]);

  return {
    filters,
    allModels,
    allCategories,
    toggleModel,
    toggleCategory,
    resetFilters,
    filteredResults,
  };
}
