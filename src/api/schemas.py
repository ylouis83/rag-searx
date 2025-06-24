"""
RAG-SearX API

TRDAPISchema
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import time


class MediaType(str, Enum):
    """"""
    TEXT_SUMMARY = "text_summary"
    TEXT_DETAIL = "text_detail" 
    IMAGE = "image"
    VIDEO_CLIP = "video_clip"


class ContextLink(BaseModel):
    """"""
    chapter_number: int = Field(..., description="")
    chapter_title: str = Field(..., description="")
    page_number: int = Field(..., description="")


class VectorDBMetadata(BaseModel):
    """Schema"""
    media_type: MediaType = Field(..., description="")
    book_title: str = Field(..., description="")
    source_text: str = Field(..., description="")
    source_url: Optional[str] = Field(None, description="/URL")
    context_link: ContextLink = Field(..., description="")


class QueryConfig(BaseModel):
    """"""
    top_k_retrieval: int = Field(default=20, ge=1, le=100, description="")
    top_k_rerank: int = Field(default=5, ge=1, le=50, description="")
    enable_multimedia: bool = Field(default=True, description="")
    enable_rerank: bool = Field(default=True, description="")
    enable_context_compression: bool = Field(default=True, description="")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="")
    
    @validator('top_k_rerank')
    def validate_rerank_size(cls, v, values):
        """"""
        if 'top_k_retrieval' in values and v > values['top_k_retrieval']:
            raise ValueError('')
        return v


class QueryRequest(BaseModel):
    """APISchema"""
    query_text: str = Field(..., min_length=1, max_length=1000, description="")
    user_id: Optional[str] = Field(None, description="ID")
    config: Optional[QueryConfig] = Field(default_factory=QueryConfig, description="")
    
    @validator('query_text')
    def validate_query_text(cls, v):
        """"""
        v = v.strip()
        if not v:
            raise ValueError('')
        return v


class RetrievedMedia(BaseModel):
    """"""
    type: str = Field(..., description=": image  video_clip")
    url: str = Field(..., description="URL")
    description: str = Field(..., description="/")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="")
    source_citation: str = Field(..., description=": ': , 45'")
    
    @validator('type')
    def validate_media_type(cls, v):
        """"""
        valid_types = ['image', 'video_clip']
        if v not in valid_types:
            raise ValueError(f': {valid_types}')
        return v


class RetrievedTextCitation(BaseModel):
    """"""
    text_snippet: str = Field(..., description="")
    source_citation: str = Field(..., description="")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="")


class DebugInfo(BaseModel):
    """"""
    query_time_ms: int = Field(..., description="()")
    llm_model_used: str = Field(..., description="LLM")
    retrieval_stats: Optional[Dict[str, Any]] = Field(None, description="")
    generation_stats: Optional[Dict[str, Any]] = Field(None, description="")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token")


class QueryResponse(BaseModel):
    """APISchema"""
    answer_text: str = Field(..., description="LLM")
    retrieved_media: List[RetrievedMedia] = Field(default=[], description="")
    retrieved_text_citations: List[RetrievedTextCitation] = Field(default=[], description="")
    debug_info: Optional[DebugInfo] = Field(None, description="")
    
    @validator('answer_text')
    def validate_answer_text(cls, v):
        """"""
        if not v.strip():
            raise ValueError('')
        return v


class HealthStatus(BaseModel):
    """"""
    status: str = Field(..., description="")
    timestamp: float = Field(default_factory=time.time, description="")
    version: str = Field(..., description="")
    dependencies: Dict[str, str] = Field(default={}, description="")


class ErrorDetail(BaseModel):
    """"""
    type: str = Field(..., description="")
    message: str = Field(..., description="")
    details: Optional[Dict[str, Any]] = Field(None, description="")


class ErrorResponse(BaseModel):
    """"""
    error: ErrorDetail = Field(..., description="")
    request_id: Optional[str] = Field(None, description="ID")
    timestamp: float = Field(default_factory=time.time, description="")


class BookInfo(BaseModel):
    """"""
    book_id: str = Field(..., description="ID")
    title: str = Field(..., description="")
    author: Optional[str] = Field(None, description="")
    language: str = Field(default="zh", description="")
    total_pages: Optional[int] = Field(None, description="")
    total_chapters: Optional[int] = Field(None, description="")
    indexed_at: float = Field(..., description="")
    file_path: str = Field(..., description="")
    file_size: int = Field(..., description="()")
    status: str = Field(..., description="")


class BookListResponse(BaseModel):
    """"""
    books: List[BookInfo] = Field(..., description="")
    total: int = Field(..., description="")


class UploadRequest(BaseModel):
    """"""
    book_title: str = Field(..., description="")
    author: Optional[str] = Field(None, description="")
    language: str = Field(default="zh", description="")
    metadata_mapping: Optional[Dict[str, Any]] = Field(None, description="")


class UploadResponse(BaseModel):
    """"""
    book_id: str = Field(..., description="ID")
    message: str = Field(..., description="")
    status: str = Field(..., description="")
    estimated_processing_time: Optional[int] = Field(None, description="()")


class IndexingProgress(BaseModel):
    """"""
    book_id: str = Field(..., description="ID")
    status: str = Field(..., description="")
    progress: float = Field(..., ge=0.0, le=1.0, description="")
    current_step: str = Field(..., description="")
    estimated_remaining_time: Optional[int] = Field(None, description="()")
    error_message: Optional[str] = Field(None, description="")


class IndexingProgressResponse(BaseModel):
    """"""
    progress: IndexingProgress = Field(..., description="")


class DeleteBookResponse(BaseModel):
    """"""
    book_id: str = Field(..., description="ID")
    message: str = Field(..., description="")
    deleted_files: List[str] = Field(default=[], description="")


class StatisticsResponse(BaseModel):
    """"""
    total_books: int = Field(..., description="")
    total_documents: int = Field(..., description="")
    total_images: int = Field(..., description="")
    total_videos: int = Field(..., description="")
    total_vectors: int = Field(..., description="")
    storage_usage: Dict[str, Any] = Field(..., description="")
    index_status: Dict[str, Any] = Field(..., description="")


# 

class ProcessingTask(BaseModel):
    """"""
    task_id: str = Field(..., description="ID")
    book_id: str = Field(..., description="ID")
    task_type: str = Field(..., description="")
    status: str = Field(..., description="")
    created_at: float = Field(default_factory=time.time, description="")
    updated_at: float = Field(default_factory=time.time, description="")
    metadata: Dict[str, Any] = Field(default={}, description="")


class ChunkData(BaseModel):
    """"""
    chunk_id: str = Field(..., description="ID")
    content: str = Field(..., description="")
    chunk_type: MediaType = Field(..., description="")
    metadata: VectorDBMetadata = Field(..., description="")
    embeddings: Optional[List[float]] = Field(None, description="")


class RetrievalResult(BaseModel):
    """"""
    chunk_data: ChunkData = Field(..., description="")
    score: float = Field(..., description="")
    rank: int = Field(..., description="")


class GenerationContext(BaseModel):
    """"""
    query: str = Field(..., description="")
    retrieved_chunks: List[RetrievalResult] = Field(..., description="")
    total_tokens: int = Field(..., description="Token")
    compressed: bool = Field(default=False, description="")
    compression_ratio: Optional[float] = Field(None, description="")


# Schema

class ModelConfig(BaseModel):
    """"""
    provider: str = Field(..., description="")
    model_name: str = Field(..., description="")
    api_key: Optional[str] = Field(None, description="API")
    base_url: Optional[str] = Field(None, description="URL")
    parameters: Dict[str, Any] = Field(default={}, description="")


class PipelineConfig(BaseModel):
    """"""
    embedding_model: ModelConfig = Field(..., description="")
    llm_model: ModelConfig = Field(..., description="LLM")
    vlm_model: ModelConfig = Field(..., description="VLM")
    rerank_model: Optional[ModelConfig] = Field(None, description="")
    retrieval_settings: QueryConfig = Field(..., description="") 