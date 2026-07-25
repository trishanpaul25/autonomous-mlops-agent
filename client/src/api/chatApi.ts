import apiClient from "./axios";
import type { ChatResponse } from "../types/pipeline";

export interface RunPipelinePayload {
  prompt: string;
  file?: File | null;
  /**
   * Generate this client-side (crypto.randomUUID()) and pass it in so you
   * can open the SSE stream at /runs/{runId}/events *before* calling this,
   * instead of only finding out the run_id once the whole pipeline is done.
   */
  runId?: string;
}

export const chatApi = {
  /**
   * POST /chat
   * multipart/form-data: `prompt` (required), `file` (optional),
   * `run_id` (optional — server generates one if omitted).
   *
   * This is a BLOCKING call from the caller's perspective — it doesn't
   * resolve until the full pipeline finishes — but the backend now runs
   * it in a worker thread, so /runs/{run_id}/events can stream progress
   * concurrently while this promise is still pending. No axios timeout is
   * set here deliberately; a long-running training job hitting a
   * client-side timeout would look like a network failure when the
   * pipeline may still be succeeding server-side.
   */
  runPipeline: ({ prompt, file, runId }: RunPipelinePayload) => {
    const form = new FormData();
    form.append("prompt", prompt);
    if (file) {
      form.append("file", file);
    }
    if (runId) {
      form.append("run_id", runId);
    }

    // IMPORTANT: apiClient (see axios.ts) sets a default
    // "Content-Type: application/json" header on the instance. That
    // default applies here too unless we explicitly clear it, which
    // stops axios/the browser from auto-generating the required
    // "multipart/form-data; boundary=..." header for this FormData body.
    // Without clearing it, the server receives a JSON content-type with a
    // multipart body, "prompt"/"run_id" never arrive as parsed Form
    // fields, and FastAPI rejects the request with 422 Unprocessable
    // Entity. (This was fixed once already — don't drop this override
    // again in a future edit to this function.)
    // Backend mounts this at prefix "/chat" + route "/", i.e. POST /chat/
    // exactly. Posting to "/chat" works today because FastAPI 307-redirects
    // it, but that's an extra round trip some proxies/environments won't
    // follow correctly for a POST — hit the real path directly instead.
    return apiClient.post<ChatResponse>("/chat/", form, {
      timeout: 0, // no client timeout — see note above
      headers: {
        "Content-Type": undefined, // let axios set multipart + boundary
      },
    });
  },
};
