"""
RAG-SearX 

AI
"""

__version__ = "1.0.0"
__author__ = "RAG-SearX Team"
__description__ = "AI"
__email__ = "team@rag-searx.com"

# 
VERSION_INFO = {
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "email": __email__,
}

# 
from .api import app
from .api.schemas import QueryRequest, QueryResponse

__all__ = [
    "app",
    "QueryRequest", 
    "QueryResponse",
    "VERSION_INFO",
] 