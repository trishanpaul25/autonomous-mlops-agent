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

@asynccontextmanager
async def lifespan(app : FastAPI):
  # The pipeline runs in a worker thread (see server/api/routes/chat.py),
  # so ProgressService needs a reference to *this* loop to publish SSE
  # events back onto it safely from that thread.
  progress_manager.bind_loop(asyncio.get_running_loop())
  logger.info(f"Starting {settings.APP_NAME} Backened")
  yield
  logger.info(f"Shutting Down {settings.APP_NAME} Backend")

app = FastAPI(
  title=settings.APP_NAME,
  version=settings.APP_VERSION,
  lifespan=lifespan
)
from fastapi.middleware.cors import CORSMiddleware
import os

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