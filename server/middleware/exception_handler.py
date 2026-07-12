from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from server.exceptions import AutonomousMLOpsException
from server.schemas import APIResponse, ErrorResponse
from server.core.constants import ErrorCode
from server.core.logging import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI):
    """
    Register all global exception handlers.
    """

    @app.exception_handler(AutonomousMLOpsException)
    async def autonomous_exception_handler(
        request: Request,
        exc: AutonomousMLOpsException,
    ):
        logger.error(
            f"{exc.error_code} : {exc.message}"
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse(
                success=False,
                error=ErrorResponse(
                    code=exc.error_code,
                    message=exc.message,
                ),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request,
        exc: Exception,
    ):
        logger.exception(exc)

        return JSONResponse(
            status_code=500,
            content=APIResponse(
                success=False,
                error=ErrorResponse(
                    code=ErrorCode.INTERNAL_SERVER_ERROR,
                    message="An unexpected error occurred.",
                ),
            ).model_dump(),
        )