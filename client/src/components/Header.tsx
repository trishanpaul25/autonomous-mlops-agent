import { useState } from "react";
import {
  LogOut,
  Home,
  Cpu,
  Info,
  Menu,
  X,
  ArrowLeft,
  LayoutDashboard,
  Rocket,
  Package,
  Activity,
  Bot,
  LogIn,
  UserPlus,
} from "lucide-react";
import { useAuth } from "../hooks/useAuth";
import { useNavigate } from "react-router-dom";

interface HeaderProps {
  isNotDashboard?: boolean;
}

const Header: React.FC<HeaderProps> = ({ isNotDashboard = false }) => {
  const { isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);

  const toggleMobileMenu = () => setMobileOpen((prev) => !prev);

  const handleBack = () => navigate(-1);

  const navigateAndClose = (path: string) => {
    navigate(path);
    setMobileOpen(false);
  };

  return (
    <header className="fixed top-0 w-full bg-[var(--color-primary)]/95 backdrop-blur-md z-50 border-b border-white/10">
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage:
            "linear-gradient(45deg, transparent 40%, rgba(255,255,255,0.05) 50%, transparent 60%)",
          backgroundSize: "25px 25px",
        }}
      />

      <nav className="relative max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
        {/* Logo */}
        <div
          onClick={() => navigate("/")}
          className="text-2xl font-bold text-[var(--color-secondary)] cursor-pointer select-none"
        >
          Auto<span className="text-blue-500">MLOps</span>
        </div>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center gap-2">
          {isAuthenticated ? (
            <>
              <button
                onClick={() => navigate("/dashboard")}
                className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/10 transition"
              >
                <LayoutDashboard className="w-4 h-4 text-blue-400" />
                ~/dashboard
              </button>

              <button
                onClick={() => navigate("/run")}
                className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/10 transition"
              >
                <Rocket className="w-4 h-4 text-green-400" />
                ~/pipeline
              </button>

              <button
                onClick={() => navigate("/deployments")}
                className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/10 transition"
              >
                <Package className="w-4 h-4 text-yellow-400" />
                ~/deployments
              </button>

              <button
                onClick={() => navigate("/monitoring")}
                className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/10 transition"
              >
                <Activity className="w-4 h-4 text-purple-400" />
                ~/monitoring
              </button>

              <button
                onClick={() => navigate("/assistant")}
                className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/10 transition"
              >
                <Bot className="w-4 h-4 text-cyan-400" />
                ~/assistant
              </button>

              {isNotDashboard && (
                <button
                  onClick={handleBack}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/10 transition"
                >
                  <ArrowLeft className="w-4 h-4 text-orange-400" />
                  back
                </button>
              )}

              <button
                onClick={logout}
                className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-red-500/20 transition"
              >
                <LogOut className="w-4 h-4 text-red-500" />
                ~/logout
              </button>
            </>
          ) : (
            <>
              <a
                onClick={() => navigate("/")}
                className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/10 transition"
              >
                <Home className="w-4 h-4 text-blue-400" />
                ~/home
              </a>

              <a
                href="#features"
                className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/10 transition"
              >
                <Cpu className="w-4 h-4 text-green-400" />
                ~/features
              </a>

              <a
                href="#about"
                className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/10 transition"
              >
                <Info className="w-4 h-4 text-purple-400" />
                ~/about
              </a>

              <button
                onClick={() => navigate("/login")}
                className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/10 transition"
              >
                <LogIn className="w-4 h-4 text-cyan-400" />
                ~/login
              </button>

              <button
                onClick={() => navigate("/register")}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white transition"
              >
                <UserPlus className="w-4 h-4" />
                Register
              </button>
            </>
          )}
        </div>

        {/* Mobile Button */}
        <button
          onClick={toggleMobileMenu}
          className="md:hidden text-[var(--color-secondary)]"
        >
          {mobileOpen ? (
            <X className="w-6 h-6" />
          ) : (
            <Menu className="w-6 h-6" />
          )}
        </button>

        {/* Mobile Menu */}
        {mobileOpen && (
          <div className="absolute top-full left-0 w-full bg-[var(--color-primary)] border-t border-white/10 flex flex-col py-4 md:hidden">
            {isAuthenticated ? (
              <>
                <button
                  onClick={() => navigateAndClose("/dashboard")}
                  className="flex items-center gap-3 px-6 py-3 hover:bg-white/10"
                >
                  <LayoutDashboard className="w-4 h-4 text-blue-400" />
                  ~/dashboard
                </button>

                <button
                  onClick={() => navigateAndClose("/run")}
                  className="flex items-center gap-3 px-6 py-3 hover:bg-white/10"
                >
                  <Rocket className="w-4 h-4 text-green-400" />
                  ~/pipeline
                </button>

                <button
                  onClick={() => navigateAndClose("/deployments")}
                  className="flex items-center gap-3 px-6 py-3 hover:bg-white/10"
                >
                  <Package className="w-4 h-4 text-yellow-400" />
                  ~/deployments
                </button>

                <button
                  onClick={() => navigateAndClose("/monitoring")}
                  className="flex items-center gap-3 px-6 py-3 hover:bg-white/10"
                >
                  <Activity className="w-4 h-4 text-purple-400" />
                  ~/monitoring
                </button>

                <button
                  onClick={() => navigateAndClose("/assistant")}
                  className="flex items-center gap-3 px-6 py-3 hover:bg-white/10"
                >
                  <Bot className="w-4 h-4 text-cyan-400" />
                  ~/assistant
                </button>

                {isNotDashboard && (
                  <button
                    onClick={() => {
                      handleBack();
                      setMobileOpen(false);
                    }}
                    className="flex items-center gap-3 px-6 py-3 hover:bg-white/10"
                  >
                    <ArrowLeft className="w-4 h-4 text-orange-400" />
                    back
                  </button>
                )}

                <button
                  onClick={() => {
                    logout();
                    setMobileOpen(false);
                  }}
                  className="flex items-center gap-3 px-6 py-3 hover:bg-red-500/20"
                >
                  <LogOut className="w-4 h-4 text-red-500" />
                  ~/logout
                </button>
              </>
            ) : (
              <>
                <a
                  href="#home"
                  className="flex items-center gap-3 px-6 py-3 hover:bg-white/10"
                >
                  <Home className="w-4 h-4 text-blue-400" />
                  ~/home
                </a>

                <a
                  href="#features"
                  className="flex items-center gap-3 px-6 py-3 hover:bg-white/10"
                >
                  <Cpu className="w-4 h-4 text-green-400" />
                  ~/features
                </a>

                <a
                  href="#about"
                  className="flex items-center gap-3 px-6 py-3 hover:bg-white/10"
                >
                  <Info className="w-4 h-4 text-purple-400" />
                  ~/about
                </a>

                <button
                  onClick={() => navigateAndClose("/login")}
                  className="flex items-center gap-3 px-6 py-3 hover:bg-white/10"
                >
                  <LogIn className="w-4 h-4 text-cyan-400" />
                  ~/login
                </button>

                <button
                  onClick={() => navigateAndClose("/register")}
                  className="flex items-center gap-3 px-6 py-3 hover:bg-blue-600/20"
                >
                  <UserPlus className="w-4 h-4 text-blue-400" />
                  Register
                </button>
              </>
            )}
          </div>
        )}
      </nav>
    </header>
  );
};

export default Header;
