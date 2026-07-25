import { useContext } from "react";
import { AuthContext } from "../context/AuthContext";
import type { AuthContextValue } from "../types/auth";

/**
 * Thin accessor for AuthContext. Throws instead of silently returning
 * undefined if used outside <AuthProvider> — fails fast at the call site
 * instead of producing a confusing "Cannot read property 'user' of
 * undefined" somewhere deep in a component.
 */
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);

  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }

  return context;
}
