import { type ReactNode } from "react";
import { ProtectedRoute } from "./ProtectedRoute";
import { Navbar } from "./Navbar";

interface ProtectedLayoutProps {
  children: ReactNode;
}

// Combines auth guarding + the shared nav, so App.tsx's route definitions
// stay flat instead of nesting <ProtectedRoute><Navbar/><Page/></ProtectedRoute>
// at every route.
export function ProtectedLayout({ children }: ProtectedLayoutProps) {
  return (
    <ProtectedRoute>
      <Navbar />
      {children}
    </ProtectedRoute>
  );
}
