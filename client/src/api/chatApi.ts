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

    // Deliberately no Content-Type header here: axios/the browser sets
    // "multipart/form-data; boundary=..." automatically for FormData
    // bodies. Setting it manually strips the boundary and breaks the
    // server's multipart parsing.
    // Backend mounts this at prefix "/chat" + route "/", i.e. POST /chat/
    // exactly. Posting to "/chat" works today because FastAPI 307-redirects
    // it, but that's an extra round trip some proxies/environments won't
    // follow correctly for a POST — hit the real path directly instead.
    return apiClient.post<ChatResponse>("/chat/", form, {
      timeout: 0, // no client timeout — see note above
    });
  },
};
