"""
RAG-SearX API

REST API
"""

import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import FileResponse

from config.settings_simple import get_settings
from .schemas import (
    QueryRequest, QueryResponse, QueryConfig,
    BookInfo, BookListResponse, UploadRequest, UploadResponse,
    DeleteBookResponse, StatisticsResponse, IndexingProgressResponse,
    DebugInfo, RetrievedMedia, RetrievedTextCitation
)
from ..utils.logger import get_logger

# 
settings = get_settings()
logger = get_logger(__name__)

# 
router = APIRouter()


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_books(request: QueryRequest) -> QueryResponse:
    """
     - 
    
    
    
    """
    start_time = time.time()
    
    try:
        logger.info(f": {request.query_text[:100]}...")
        
        # 
        from ..core.query_engine import process_query
        
        try:
            # 
            response = await process_query(request)
            return response
        except Exception as query_error:
            logger.warning(f": {query_error}")
            # 
            answer_text = f": '{request.query_text}'"
        
        retrieved_media = [
            RetrievedMedia(
                type="image",
                url="/media/images/sample.jpg",
                description="",
                relevance_score=0.85,
                source_citation=": , 10"
            )
        ]
        
        retrieved_text_citations = [
            RetrievedTextCitation(
                text_snippet="...",
                source_citation=", 10",
                relevance_score=0.92
            )
        ]
        
        query_time_ms = int((time.time() - start_time) * 1000)
        
        debug_info = DebugInfo(
            query_time_ms=query_time_ms,
            llm_model_used=settings.llm.model,
            retrieval_stats={
                "total_retrieved": 20,
                "after_rerank": 5,
                "compression_applied": False
            },
            generation_stats={
                "input_tokens": 1500,
                "output_tokens": 300,
                "total_tokens": 1800
            }
        )
        
        return QueryResponse(
            answer_text=answer_text,
            retrieved_media=retrieved_media,
            retrieved_text_citations=retrieved_text_citations,
            debug_info=debug_info
        )
        
    except Exception as e:
        logger.error(f": {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")


@router.get("/books", response_model=BookListResponse, tags=["Books"])
async def list_books() -> BookListResponse:
    """"""
    
    try:
        logger.info("")
        
        # TODO: 
        # books = await get_books_from_db()
        
        # 
        books = [
            BookInfo(
                book_id="book_001",
                title="",
                author="·",
                language="zh",
                total_pages=350,
                total_chapters=20,
                indexed_at=time.time(),
                file_path="/data/books/hundred_years_of_solitude.pdf",
                file_size=5242880,
                status="indexed"
            )
        ]
        
        return BookListResponse(
            books=books,
            total=len(books)
        )
        
    except Exception as e:
        logger.error(f": {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")


@router.post("/books/upload", response_model=UploadResponse, tags=["Books"])
async def upload_book(
    background_tasks: BackgroundTasks,
    book_file: UploadFile = File(...),
    book_title: str = Form(...),
    author: Optional[str] = Form(None),
    language: str = Form("zh"),
    media_files: Optional[List[UploadFile]] = File(None)
) -> UploadResponse:
    """
    
    
    PDFePub
    
    """
    
    try:
        logger.info(f": {book_title}")
        
        # 
        if not book_file.filename.lower().endswith(('.pdf', '.epub', '.txt')):
            raise HTTPException(
                status_code=400, 
                detail="PDFePubTXT"
            )
        
        # ID
        book_id = f"book_{int(time.time())}"
        
        # TODO: 
        # 1. 
        # await save_uploaded_files(book_file, media_files, book_id)
        
        # 2. 
        # background_tasks.add_task(start_indexing_task, book_id, book_title, author, language)
        
        logger.info(f": {book_id}")
        
        return UploadResponse(
            book_id=book_id,
            message="",
            status="processing",
            estimated_processing_time=600  # 10
        )
        
    except Exception as e:
        logger.error(f": {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")


@router.get("/books/{book_id}/progress", response_model=IndexingProgressResponse, tags=["Books"])
async def get_indexing_progress(book_id: str) -> IndexingProgressResponse:
    """"""
    
    try:
        logger.info(f": {book_id}")
        
        # TODO: 
        # progress = await get_indexing_progress_from_db(book_id)
        
        # 
        from .schemas import IndexingProgress
        
        progress = IndexingProgress(
            book_id=book_id,
            status="processing",
            progress=0.65,
            current_step="...",
            estimated_remaining_time=180
        )
        
        return IndexingProgressResponse(progress=progress)
        
    except Exception as e:
        logger.error(f": {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")


@router.delete("/books/{book_id}", response_model=DeleteBookResponse, tags=["Books"])
async def delete_book(book_id: str) -> DeleteBookResponse:
    """"""
    
    try:
        logger.info(f": {book_id}")
        
        # TODO: 
        # 1. 
        # await delete_vectors_from_db(book_id)
        
        # 2. 
        # deleted_files = await delete_book_files(book_id)
        
        # 3. 
        # await delete_book_from_db(book_id)
        
        logger.info(f": {book_id}")
        
        return DeleteBookResponse(
            book_id=book_id,
            message="",
            deleted_files=[
                f"/data/books/{book_id}.pdf",
                f"/data/media/{book_id}/",
                f"/data/vectors/{book_id}/"
            ]
        )
        
    except Exception as e:
        logger.error(f": {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")


@router.get("/status", tags=["System"])
async def get_system_status():
    """"""
    
    try:
        logger.info("")
        
        # 
        vector_db_connected = True
        llm_available = True
        embedding_model_loaded = True
        
        try:
            from ..vectorstore.milvus_store import get_milvus_store
            vector_store = await get_milvus_store()
        except Exception:
            vector_db_connected = False
        
        try:
            from ..generator.dashscope_llm import get_dashscope_llm
            llm = get_dashscope_llm()
        except Exception:
            llm_available = False
        
        try:
            from ..generator.embedding_models import get_embedding_model
            embedding_model = get_embedding_model()
        except Exception:
            embedding_model_loaded = False
        
        return {
            "vector_db_connected": vector_db_connected,
            "llm_available": llm_available,
            "embedding_model_loaded": embedding_model_loaded,
            "indexed_books_count": 0,  # TODO: 
            "total_chunks": 0  # TODO: 
        }
        
    except Exception as e:
        logger.error(f": {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")


@router.get("/statistics", response_model=StatisticsResponse, tags=["Statistics"])
async def get_statistics() -> StatisticsResponse:
    """"""
    
    try:
        logger.info("")
        
        # TODO: 
        # stats = await get_system_statistics()
        
        # 
        return StatisticsResponse(
            total_books=3,
            total_documents=250,
            total_images=45,
            total_videos=12,
            total_vectors=15000,
            storage_usage={
                "books": "156 MB",
                "media": "2.3 GB", 
                "vectors": "45 MB",
                "total": "2.5 GB"
            },
            index_status={
                "healthy_indices": 3,
                "failed_indices": 0,
                "processing_indices": 0
            }
        )
        
    except Exception as e:
        logger.error(f": {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")


@router.get("/media/{file_path:path}", tags=["Media"])
async def serve_media(file_path: str):
    """"""
    
    try:
        # 
        full_path = settings.storage.storage_path / "media" / file_path
        
        # 
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="")
        
        # 
        if not str(full_path).startswith(str(settings.storage.storage_path / "media")):
            raise HTTPException(status_code=403, detail="")
        
        return FileResponse(
            path=str(full_path),
            filename=full_path.name
        )
        
    except Exception as e:
        logger.error(f": {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")


# 

@router.post("/dev/reindex/{book_id}", tags=["Development"])
async def reindex_book(book_id: str, background_tasks: BackgroundTasks):
    """"""
    
    if not settings.development.debug:
        raise HTTPException(status_code=404, detail="")
    
    try:
        logger.info(f": {book_id}")
        
        # TODO: 
        # background_tasks.add_task(reindex_book_task, book_id)
        
        return {"message": f": {book_id}"}
        
    except Exception as e:
        logger.error(f": {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")


@router.get("/dev/config", tags=["Development"])
async def get_config():
    """"""
    
    if not settings.development.debug:
        raise HTTPException(status_code=404, detail="")
    
    # 
    safe_config = {
        "api": {
            "title": settings.api.title,
            "version": settings.api.version,
            "host": settings.api.host,
            "port": settings.api.port,
        },
        "vector_db": {
            "type": settings.vector_db.type,
            "host": settings.vector_db.host,
            "port": settings.vector_db.port,
        },
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
        },
        "embedding": {
            "provider": settings.embedding.provider,
            "model": settings.embedding.model,
            "dimension": settings.embedding.dimension,
        }
    }
    
    return safe_config 