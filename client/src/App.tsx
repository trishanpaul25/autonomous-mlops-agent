import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthProvider";
import { ProtectedLayout } from "./components/ProtectedLayout";
import { Login } from "./pages/Login/Login";
import { Register } from "./pages/Register/Register";
import { Dashboard } from "./pages/Dashboard/Dashboard";
import { DatasetUpload } from "./pages/DatasetUpload/DatasetUpload";
import { Deployments } from "./pages/Deployments/Deployments";
import { DeploymentDetail } from "./pages/Deployments/DeploymentDetail";
import HeroSection from "./components/HeroSection";

export function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/" element={<HeroSection />} />

          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          <Route
            path="/dashboard"
            element={
              <ProtectedLayout>
                <Dashboard />
              </ProtectedLayout>
            }
          />

          <Route
            path="/run"
            element={
              <ProtectedLayout>
                <DatasetUpload />
              </ProtectedLayout>
            }
          />

          <Route
            path="/deployments"
            element={
              <ProtectedLayout>
                <Deployments />
              </ProtectedLayout>
            }
          />

          <Route
            path="/deployments/:deploymentId"
            element={
              <ProtectedLayout>
                <DeploymentDetail />
              </ProtectedLayout>
            }
          />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
