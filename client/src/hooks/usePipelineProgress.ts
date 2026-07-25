import { useEffect, useRef, useState } from "react";
import { tokenStorage } from "../api/axios";
import type { ProgressEvent } from "../types/pipeline";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface UsePipelineProgressOptions {
  runId: string | null;
  enabled: boolean;
}

interface UsePipelineProgressResult {
  events: ProgressEvent[];
  isConnected: boolean;
}

/**
 * Subscribes to GET /runs/{run_id}/events (Server-Sent Events) for live
 * pipeline progress.
 *
 * IMPORTANT — this is currently a dead end in practice: `run_id` is
 * generated server-side inside the POST /chat handler and only reaches the
 * client in the final ChatResponse, by which point the run is already
 * over. There is no point in the current flow where we have a `runId`
 * *and* the pipeline is still in progress, so `enabled` should stay false
 * until the backend either (a) accepts a client-generated run_id on
 * /chat, or (b) exposes a separate "start run" endpoint that returns
 * run_id immediately and runs the pipeline in the background.
 *
 * Wired up now, disabled by default, so flipping it on later is a
 * one-line change in DatasetUpload.tsx rather than new plumbing.
 */
export function usePipelineProgress({
  runId,
  enabled,
}: UsePipelineProgressOptions): UsePipelineProgressResult {
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!enabled || !runId) {
      return;
    }

    // EventSource can't set an Authorization header, so the token has to
    // ride along as a query param if/when this endpoint is protected.
    // /runs/{run_id}/events isn't behind get_current_user today, but this
    // keeps the hook correct if that changes.
    const token = tokenStorage.get();
    const url = new URL(`${API_BASE_URL}/runs/${runId}/events`);
    if (token) {
      url.searchParams.set("token", token);
    }

    const source = new EventSource(url.toString());
    sourceRef.current = source;

    source.onopen = () => setIsConnected(true);

    source.onerror = () => {
      setIsConnected(false);
      source.close();
    };

    // Backend emits named events (event: info/step/complete/...), so we
    // listen on the generic message channel plus each known type to
    // capture all of them regardless of which `event:` field was sent.
    const handleEvent = (e: MessageEvent) => {
      try {
        const parsed: ProgressEvent = JSON.parse(e.data);
        setEvents((prev) => [...prev, parsed]);
      } catch {
        // Malformed event — skip rather than crash the stream.
      }
    };

    const eventTypes: ProgressEvent["type"][] = [
      "info",
      "success",
      "warning",
      "error",
      "thinking",
      "step",
      "complete",
    ];
    eventTypes.forEach((type) => source.addEventListener(type, handleEvent));

    return () => {
      eventTypes.forEach((type) =>
        source.removeEventListener(type, handleEvent),
      );
      source.close();
      sourceRef.current = null;
      setIsConnected(false);
    };
  }, [runId, enabled]);

  return { events, isConnected };
}
