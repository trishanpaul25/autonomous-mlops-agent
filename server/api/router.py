from fastapi import APIRouter
from server.api.routes.health import router as health_router
from server.api.routes.chat import router as chat_router


api_router = APIRouter()

api_router.include_router(chat_router)

api_router.include_router(health_router)