import axios, { type AxiosError, type InternalAxiosRequestConfig } from "axios";
import type { ApiErrorShape } from "../types/auth";

// No /api prefix on this backend — routes are mounted at root (/auth/login, /auth/me, ...)
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const TOKEN_KEY = "mlops_token";

export const tokenStorage = {
  get: (): string | null => localStorage.getItem(TOKEN_KEY),
  set: (token: string): void => localStorage.setItem(TOKEN_KEY, token),
  clear: (): void => localStorage.removeItem(TOKEN_KEY),
};

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Attach bearer token automatically
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = tokenStorage.get();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error),
);

// Callback the AuthContext registers so we can force a logout on 401
// without this file needing to import React/context directly.
let onUnauthorized: (() => void) | null = null;
export const registerUnauthorizedHandler = (handler: () => void): void => {
  onUnauthorized = handler;
};

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiErrorShape>) => {
    if (error.response?.status === 401) {
      // Backend has no refresh-token endpoint today. If one is added later,
      // this is the single place to slot in a refresh-and-retry attempt
      // before falling back to logout.
      tokenStorage.clear();
      onUnauthorized?.();
    }
    return Promise.reject(error);
  },
);

// Normalizes FastAPI's error shape into a plain string. `detail` is a
// string for your own `HTTPException(detail="...")` calls, but for
// automatic 422 validation errors FastAPI sends an ARRAY of
// { type, loc, msg, input } objects instead — rendering that array
// directly as JSX children crashes React ("Objects are not valid as a
// React child"), so it has to be flattened to text here.
export const extractErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError<ApiErrorShape>(error)) {
    const detail = error.response?.data?.detail;

    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            const field = Array.isArray(item.loc)
              ? item.loc[item.loc.length - 1]
              : undefined;
            return field ? `${field}: ${item.msg}` : String(item.msg);
          }
          return typeof item === "string" ? item : JSON.stringify(item);
        })
        .join("; ");
    }

    if (typeof detail === "string" && detail) {
      return detail;
    }

    return error.message || "Something went wrong.";
  }
  return "Something went wrong.";
};

export default apiClient;
