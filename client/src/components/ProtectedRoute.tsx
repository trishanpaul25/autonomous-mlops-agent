import { type ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

interface ProtectedRouteProps {
  children: ReactNode;
}

/**
 * Guards routes that require an authenticated session.
 *
 * Three states matter here:
 * 1. isLoading true  -> we're still validating a stored token against
 *    GET /auth/me (see AuthProvider's bootstrap effect). Render nothing
 *    rather than redirecting, otherwise a valid session on refresh would
 *    flash to /login and then bounce back.
 * 2. isLoading false, not authenticated -> redirect to /login, remembering
 *    where they were trying to go so Login can send them back after.
 * 3. isLoading false, authenticated -> render the protected content.
 */
export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="auth-boot-screen" role="status" aria-live="polite">
        <div className="auth-boot-spinner" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
}
