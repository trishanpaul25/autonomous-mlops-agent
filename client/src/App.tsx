import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthProvider";
import { ProtectedLayout } from "./components/ProtectedLayout";
import { Login } from "./pages/Login/Login";
import { Register } from "./pages/Register/Register";
import { Dashboard } from "./pages/Dashboard/Dashboard";
import { DatasetUpload } from "./pages/DatasetUpload/DatasetUpload";

export function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
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

          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
