"""
RAG-SearX FastAPI

API
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from config.settings_simple import get_settings
from .routes import router
from .schemas import ErrorResponse, ErrorDetail
from ..utils.logger import get_logger

# 
settings = get_settings()
logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """ID - ID"""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # ID
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # 
        if process_time > 5.0:  # 5
            logger.warning(
                f"Slow request: {request.method} {request.url} "
                f"took {process_time:.2f}s"
            )
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """"""
    
    # 
    logger.info("RAG-SearX API starting up...")
    
    try:
        # 
        # await init_vector_db()
        
        # 
        # await init_cache()
        
        # 
        # await init_models()
        
        logger.info("RAG-SearX API startup complete")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start RAG-SearX API: {e}")
        raise
    finally:
        # 
        logger.info("RAG-SearX API shutting down...")
        
        # 
        # await cleanup_resources()
        
        logger.info("RAG-SearX API shutdown complete")


def create_app() -> FastAPI:
    """FastAPI"""
    
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=lifespan,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZIP
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(TimingMiddleware)
    
    # 
    app.include_router(router, prefix="/api")
    
    return app


# 
app = create_app()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP"""
    
    request_id = getattr(request.state, 'request_id', None)
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            type="HTTP_ERROR",
            message=exc.detail,
            details={
                "status_code": exc.status_code,
                "path": str(request.url),
                "method": request.method,
            }
        ),
        request_id=request_id,
    )
    
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail} "
        f"[{request.method} {request.url}] [RequestID: {request_id}]"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """"""
    
    request_id = getattr(request.state, 'request_id', None)
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            type="VALIDATION_ERROR",
            message="",
            details={
                "validation_errors": exc.errors(),
                "path": str(request.url),
                "method": request.method,
            }
        ),
        request_id=request_id,
    )
    
    logger.warning(
        f"Validation error: {exc.errors()} "
        f"[{request.method} {request.url}] [RequestID: {request_id}]"
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response.dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """"""
    
    request_id = getattr(request.state, 'request_id', None)
    
    # 
    logger.error(
        f"Unhandled exception: {str(exc)} "
        f"[{request.method} {request.url}] [RequestID: {request_id}]",
        exc_info=True
    )
    
    # 
    if settings.enable_verbose_errors:
        details = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "path": str(request.url),
            "method": request.method,
        }
    else:
        details = {
            "path": str(request.url),
            "method": request.method,
        }
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            type="INTERNAL_ERROR",
            message="" if not settings.enable_verbose_errors else str(exc),
            details=details
        ),
        request_id=request_id,
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict(),
    )


@app.get("/", include_in_schema=False)
async def root():
    """ - API"""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs_url": "/docs" if settings.enable_docs else None,
        "status": "healthy",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """"""
    from .schemas import HealthStatus
    
    # TODO: 
    dependencies = {
        "vector_db": "healthy",
        "cache": "healthy", 
        "llm": "healthy",
        "embedding": "healthy",
    }
    
    return HealthStatus(
        status="healthy",
        version=settings.api_version,
        dependencies=dependencies,
    )


if __name__ == "__main__":
    # 
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info",
    ) 