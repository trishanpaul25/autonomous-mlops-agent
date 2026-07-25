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
 * Callers pass a client-generated run_id (crypto.randomUUID()) alongside
 * that same run_id in the POST /chat form data, and set `enabled` true as
 * soon as the request is fired — see DatasetUpload.tsx. The backend now
 * runs the pipeline in a worker thread (server/api/routes/chat.py), so
 * this stream can deliver events concurrently while /chat is still
 * in flight, rather than only after it resolves.
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

    source.onopen = () => {
      // Clear the previous run's log here, inside the callback that fires
      // once the connection is actually live — not synchronously in the
      // effect body. onopen always fires before any message event per the
      // SSE spec, so this can't race with handleEvent below.
      setEvents([]);
      setIsConnected(true);
    };

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
