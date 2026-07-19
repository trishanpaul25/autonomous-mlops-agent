from fastapi import APIRouter
from server.api.routes.health import router as health_router
from server.api.routes.chat import router as chat_router
from server.api.routes.upload import router as upload_router
from server.api.routes.runs import router as runs_router

api_router = APIRouter()

api_router.include_router(chat_router)
api_router.include_router(upload_router)
api_router.include_router(runs_router)
api_router.include_router(health_router)
