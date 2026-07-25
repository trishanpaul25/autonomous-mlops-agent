import os
from fastapi import FastAPI
from server.api.router import api_router
from server.core.config import settings
from server.core.logging import initialize_logging, get_logger
from server.middleware import register_exception_handlers
from contextlib import asynccontextmanager
import asyncio
from server.services.progress_manager import progress_manager
"""
  This ensures logging is initialized exactly once during application startup and gives you a clean place to initialize other resources later (database connections, MLflow, Redis, etc.)
"""
initialize_logging()
logger = get_logger(__name__)
def _reload_latest_deployment() -> None:
  """
  Repopulate the in-process ModelServerRegistry on startup.

  ModelServerRegistry (tools/deployment/model_server_registry.py) is a
  plain in-memory dict — it's wiped every time this process restarts,
  even though the underlying model is safely persisted in MLflow
  (MLFLOW_TRACKING_URI) and the deployment record is safely persisted
  in Postgres. Without this, every restart means /predict/{run_id}
  404s until someone re-runs the whole pipeline.

  "One live model at a time" demo model: on boot, look up the most
  recently completed deployment and load *only* that one model back
  into memory, keyed by its run_id (matching what DeploymentAgent
  does at the end of a normal pipeline run).
  """
  try:
    import mlflow
    from server.db.database import SessionLocal
    from server.repositories.deployment_repository import DeploymentRepository
    from server.repositories.model_registry_repository import ModelRegistryRepository
    from tools.deployment.model_server_registry import ModelServerRegistry

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
      mlflow.set_tracking_uri(tracking_uri)

    db = SessionLocal()
    try:
      deployment = DeploymentRepository(db).get_latest_completed()
      if deployment is None:
        logger.info("[Startup] No prior completed deployment found — nothing to preload.")
        return

      registry_row = ModelRegistryRepository(db).get_by_run_id(deployment.run_id)
      if registry_row is None or not registry_row.mlflow_run_id:
        logger.warning(
          "[Startup] Deployment %s has no matching model_registry row with "
          "an mlflow_run_id — cannot reload its model.", deployment.run_id,
        )
        return

      model_uri = f"runs:/{registry_row.mlflow_run_id}/model"
      model = mlflow.pyfunc.load_model(model_uri)

      deployment_id = str(deployment.run_id)
      ModelServerRegistry.register(deployment_id, model)
      logger.info(
        "[Startup] Reloaded '%s' into ModelServerRegistry under deployment_id=%s",
        model_uri, deployment_id,
      )
    finally:
      db.close()
  except Exception as exc:  # noqa: BLE001
    # Never let a preload failure stop the server from starting —
    # it just means /predict/{run_id} will 404 until a fresh pipeline
    # run deploys a model again, same as today's behavior.
    logger.warning("[Startup] Failed to reload latest deployment: %s", exc)
  

@asynccontextmanager
async def lifespan(app : FastAPI):
  # The pipeline runs in a worker thread (see server/api/routes/chat.py),
  # so ProgressService needs a reference to *this* loop to publish SSE
  # events back onto it safely from that thread.
  progress_manager.bind_loop(asyncio.get_running_loop())
  logger.info(f"Starting {settings.APP_NAME} Backened")
  _reload_latest_deployment() 
  
  yield
  logger.info(f"Shutting Down {settings.APP_NAME} Backend")

app = FastAPI(
  title=settings.APP_NAME,
  version=settings.APP_VERSION,
  lifespan=lifespan
)
from fastapi.middleware.cors import CORSMiddleware


# Local dev origin is always allowed. In production, set ALLOWED_ORIGINS
# on Render to a comma-separated list, e.g.
# ALLOWED_ORIGINS=https://your-app.vercel.app,https://your-custom-domain.com
_default_origins = ["http://localhost:5173"]
_env_origins = os.getenv("ALLOWED_ORIGINS", "")
allow_origins = _default_origins + [o.strip() for o in _env_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#from the middleware to register all global exception handlers
register_exception_handlers(app)

app.include_router(api_router)

@app.get("/")
def root():
  logger.info("Root endpoint accessed")
  return {"message": "Autonomous MLOps AI Backend Running"}

from server.exceptions import DatasetNotFoundException

"""
middleware exception testing
@app.get("/test-error")
def test():
    raise DatasetNotFoundException()

@app.get("/test-crash")
def crash():
    x = 1 / 0

"""