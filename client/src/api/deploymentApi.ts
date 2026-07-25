import apiClient from "./axios";
import type {
  DeploymentSummary,
  DeploymentDetail,
  PredictResponse,
} from "../types/deployment";

export const deploymentApi = {
  /** GET /deployments/ — every deployment the current user has created. */
  list: () => apiClient.get<DeploymentSummary[]>("/deployments/"),

  /**
   * GET /deployments/{deploymentId} — single deployment plus its latest
   * monitoring snapshot. `deploymentId` is the run_id (same value used
   * in predict()), not the internal deployments.id.
   */
  get: (deploymentId: string) =>
    apiClient.get<DeploymentDetail>(`/deployments/${deploymentId}`),

  /**
   * POST /predict/{deploymentId} — score raw input rows against the
   * deployed model. No client-side timeout: inference is normally fast,
   * but a cold model server restart shouldn't look like a network error.
   */
  predict: (deploymentId: string, records: Record<string, unknown>[]) =>
    apiClient.post<PredictResponse>(
      `/predict/${deploymentId}`,
      { records },
      { timeout: 0 },
    ),
};
