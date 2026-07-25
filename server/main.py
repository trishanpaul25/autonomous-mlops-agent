from fastapi import FastAPI
from server.api.router import api_router
from server.core.config import settings
from server.core.logging import initialize_logging, get_logger
from server.middleware import register_exception_handlers
from contextlib import asynccontextmanager
"""
  This ensures logging is initialized exactly once during application startup and gives you a clean place to initialize other resources later (database connections, MLflow, Redis, etc.)
"""
initialize_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app : FastAPI):
  logger.info(f"Starting {settings.APP_NAME} Backened")
  yield
  logger.info(f"Shutting Down {settings.APP_NAME} Backend")

app = FastAPI(
  title=settings.APP_NAME,
  version=settings.APP_VERSION,
  lifespan=lifespan
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # your Vite dev server
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