// Mirrors server/schemas/chat.py exactly.

export interface ModelSelectionResult {
  task_type: string | null;
  primary_model_name: string | null;
  primary_model_library: string | null;
  ranking: string[];
  reasoning: string | null;
  confidence: number;
  summary: string | null;
}

export interface ModelTrainingResult {
  training_status: string | null;
  train_samples: number;
  test_samples: number;
  trained_models: Record<string, unknown>[];
  failed_models: Record<string, unknown>[];
  total_execution_time_seconds: number;
  summary: string | null;
}

export interface HyperparameterOptimizationResult {
  optimization_status: string | null;
  scoring_metric: string | null;
  best_overall_model_name: string | null;
  best_overall_score: number;
  optimized_models: Record<string, unknown>[];
  failed_models: Record<string, unknown>[];
  total_execution_time_seconds: number;
  summary: string | null;
}

export interface ModelEvaluationResult {
  evaluation_status: string | null;
  primary_metric: string | null;
  best_model_name: string | null;
  best_model_metrics: Record<string, number>;
  comparison_table: Record<string, unknown>[];
  summary: string | null;
}

// POST /chat response. Note: this only arrives once the ENTIRE pipeline
// (ingestion -> ... -> deployment) has finished — /chat blocks for the
// full run, there's no partial/streaming version of this object.
export interface ChatResponse {
  user_prompt: string;
  assistant_message: string | null;

  run_id: string;
  status: string;
  execution_time: number | null;
  completed_steps: string[];
  logs: string[];

  dataset_name: string | null;
  rows: number | null;
  columns: number | null;

  problem_type: string | null;
  target_column: string | null;

  model_selection: ModelSelectionResult | null;
  model_training: ModelTrainingResult | null;
  hyperparameter_optimization: HyperparameterOptimizationResult | null;
  model_evaluation: ModelEvaluationResult | null;

  mlflow_run_id: string | null;
  model_name: string | null;
  model_path: string | null;
  metrics: Record<string, unknown>;
}

// Mirrors server/services/progress_types.py + events.py.
// Used by the SSE hook — currently unreachable in practice (see
// usePipelineProgress.ts) since run_id isn't known until /chat resolves,
// but typed now so wiring it up later is a one-line change, not a rewrite.
export type ProgressEventType =
  | "info"
  | "success"
  | "warning"
  | "error"
  | "thinking"
  | "step"
  | "complete";

export interface ProgressEvent {
  run_id: string;
  type: ProgressEventType;
  message: string;
  timestamp: string;
}
