import { Terminal, GitBranch } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useContext } from "react";
import { AuthContext } from "../context/AuthContext";
import Header from "./Header";
import { toast } from "sonner";

const HeroSection: React.FC = () => {
  const navigate = useNavigate();
  const auth = useContext(AuthContext);

  if (!auth) {
    throw new Error("HeroSection must be used within AuthProvider");
  }

  const { isAuthenticated } = auth;
  return (
    <section
      id="home"
      className="w-full min-h-screen flex items-center justify-center relative overflow-hidden bg-[var(--color-primary)]"
    >
      <Header />
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16 lg:py-20">
        <div className="text-center relative z-10">
          {/* Terminal Window */}
          <div className="mb-8 sm:mb-12 mx-auto max-w-2xl">
            <div className="bg-gray-900 rounded-lg border border-gray-700 shadow-2xl">
              <div className="flex items-center justify-between px-3 sm:px-4 py-2 sm:py-3 bg-gray-800 rounded-t-lg border-b border-gray-700">
                <div className="flex space-x-1.5 sm:space-x-2">
                  <div className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-red-500" />
                  <div className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-yellow-500" />
                  <div className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-green-500" />
                </div>

                <div className="text-gray-400 text-xs sm:text-sm spline-sans-mono-400">
                  MLOps Console
                </div>

                <div className="w-8 sm:w-12" />
              </div>

              <div className="p-4 sm:p-6 spline-sans-mono-400 text-left text-sm sm:text-base space-y-2">
                <div className="text-green-400">
                  ✓ Dataset validated successfully
                </div>
                <div className="text-blue-400">
                  ✓ Model training pipeline initialized
                </div>
                <div className="text-purple-400">
                  ✓ Hyperparameter optimization complete
                </div>
                <div className="text-yellow-400">
                  ✓ Deployment endpoint ready
                </div>
              </div>
            </div>
          </div>

          <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-7xl spline-sans-mono-400 font-bold mb-4 sm:mb-6 text-black px-4">
            Autonomous <span className="text-blue-400">MLOps</span>
          </h1>

          <h2 className="text-xl sm:text-2xl md:text-3xl spline-sans-mono-400 font-medium mb-6 sm:mb-8 text-gray-400 px-4">
            Build, Train & <span className="text-green-400">Deploy</span> AI
            Models
          </h2>

          <p className="text-base sm:text-lg text-gray-800 mb-8 sm:mb-12 max-w-3xl mx-auto spline-sans-mono-400 leading-relaxed px-4">
            End-to-end autonomous machine learning powered by{" "}
            <span className="text-blue-400">AI agents</span>.
            <br className="hidden sm:block" />
            <span className="sm:hidden"> </span>
            Automate <span className="text-green-400">
              data validation
            </span>,{" "}
            <span className="text-purple-400">feature engineering</span>,{" "}
            <span className="text-yellow-400">model optimization</span>, and{" "}
            <span className="text-red-400">deployment</span> with a single
            workflow.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center spline-sans-mono-400 px-4">
            <button
              onClick={() => {
                if (!isAuthenticated) {
                  toast.warning("Please login first to launch the pipeline.");

                  navigate("/login", {
                    state: { from: "/run" },
                  });

                  return;
                }

                navigate("/run");
              }}
              className="w-full sm:w-auto px-6 sm:px-8 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg hover:cursor-pointer text-white font-semibold transition-all duration-200 transform hover:scale-105 border border-blue-500"
            >
              <Terminal className="w-4 h-4 sm:w-5 sm:h-5 inline mr-2" />
              Get Started
            </button>

            <button
              onClick={() => {
                document.querySelector("#features")?.scrollIntoView({
                  behavior: "smooth",
                });
              }}
              className="w-full sm:w-auto px-6 sm:px-8 py-3 border border-gray-600 text-gray-700 hover:text-gray-300 hover:border-gray-500 hover:cursor-pointer rounded-lg font-semibold transition-all duration-200"
            >
              <GitBranch className="w-4 h-4 sm:w-5 sm:h-5 inline mr-2" />
              Explore Features
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
