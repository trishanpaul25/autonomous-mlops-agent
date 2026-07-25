import { NavLink } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import "./Navbar.css";

const NAV_LINKS = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/run", label: "Run Pipeline" },
  { to: "/deployments", label: "Deployments" },
];

export function Navbar() {
  const { user, logout } = useAuth();

  return (
    <nav className="navbar">
      <div className="navbar-left">
        <span className="auth-logo">MLOps</span>
        <div className="navbar-links">
          {NAV_LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) =>
                isActive ? "navbar-link navbar-link-active" : "navbar-link"
              }
            >
              {link.label}
            </NavLink>
          ))}
        </div>
      </div>

      <div className="navbar-right">
        <span className="navbar-user">{user?.username}</span>
        <button className="logout-button" onClick={logout}>
          Sign out
        </button>
      </div>
    </nav>
  );
}
