"""
RAG-SearX API

FastAPI
"""

__version__ = "1.0.0"
__author__ = "RAG-SearX Team"
__description__ = "AIAPI"

from .main import app
from .schemas import *
from .routes import router

__all__ = [
    "app",
    "router",
    # Schemas
    "QueryRequest",
    "QueryResponse", 
    "QueryConfig",
    "RetrievedMedia",
    "RetrievedTextCitation",
    "DebugInfo",
    "HealthStatus",
    "ErrorResponse",
    "BookInfo",
    "BookListResponse",
    "UploadRequest",
    "UploadResponse",
    "VectorDBMetadata",
    "MediaType",
    "ContextLink",
] 