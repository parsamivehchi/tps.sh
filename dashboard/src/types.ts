export interface DashboardData {
  meta: RunMeta;
  models: Record<string, ModelInfo>;
  categories: Record<string, string>;
  results: TestResult[];
  rankings: Rankings;
}

export interface RunMeta {
  run_id: string;
  started_at: string;
  completed_at: string;
  total_tests: number;
  model_count: number;
}

export interface ModelInfo {
  name: string;
  provider: string;
  model_type: string;
  cost_input: number;
  cost_output: number;
}

export interface Scores {
  correctness: number;
  completeness: number;
  clarity: number;
  weighted: number;
  reasoning: string;
}

export interface TestResult {
  model_name: string;
  prompt_id: string;
  category: string;
  output: string;
  ttft_ms: number;
  total_time_ms: number;
  tokens_per_sec: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  memory_mb: number | null;
  error: string | null;
  scores?: Scores | null;
  self_bias_flag?: boolean;
  judge_model?: string;
}

export interface Rankings {
  fastest_ttft: RankEntry | null;
  highest_tps: RankEntry | null;
  lowest_cost: RankEntry | null;
  best_quality: RankEntry | null;
  best_value: RankEntry | null;
  category_bests: Record<string, CategoryBest>;
}

export interface RankEntry {
  model_name: string;
  ttft_ms?: number;
  tokens_per_sec?: number;
  cost_usd?: number;
  score_weighted?: number;
}

export interface CategoryBest {
  label: string;
  best_model: string;
  score: number;
}

export type FilterState = {
  models: string[];
  categories: string[];
};
