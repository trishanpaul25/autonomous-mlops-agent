import { useAuth } from "../../hooks/useAuth";
import "./Dashboard.css";

// Header/logout now live in Navbar (shared across all protected pages) —
// this file only owns the dashboard content itself.
export function Dashboard() {
  const { user } = useAuth();

  return (
    <div className="dashboard-page">
      <main className="dashboard-content">
        <h1>Welcome, {user?.username}</h1>
        <p className="dashboard-subtitle">{user?.email}</p>

        <div className="dashboard-stats">
          <div className="stat-card">
            <span className="stat-value">{user?.total_runs ?? 0}</span>
            <span className="stat-label">Total Runs</span>
          </div>
          <div className="stat-card">
            <span className="stat-value stat-success">
              {user?.successful_runs ?? 0}
            </span>
            <span className="stat-label">Successful</span>
          </div>
          <div className="stat-card">
            <span className="stat-value stat-error">
              {user?.failed_runs ?? 0}
            </span>
            <span className="stat-label">Failed</span>
          </div>
        </div>
      </main>
    </div>
  );
}
