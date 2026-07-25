import apiClient from "./axios";
import type { ChatResponse } from "../types/pipeline";

export interface RunPipelinePayload {
  prompt: string;
  file?: File | null;
}

export const chatApi = {
  /**
   * POST /chat
   * multipart/form-data: `prompt` (Form field, required) + `file` (optional).
   *
   * This is a BLOCKING call on the backend — it runs the full LangGraph
   * pipeline (ingestion through deployment) inside the request handler and
   * only responds once everything finishes. No axios timeout is set here
   * deliberately; a long-running training job hitting a client-side
   * timeout would look like a network failure when the pipeline may still
   * be succeeding server-side.
   */
  runPipeline: ({ prompt, file }: RunPipelinePayload) => {
    const form = new FormData();
    form.append("prompt", prompt);
    if (file) {
      form.append("file", file);
    }

    return apiClient.post<ChatResponse>("/chat", form, {
      timeout: 0, // no client timeout — see note above
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  },
};
