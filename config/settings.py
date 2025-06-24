"""
RAG-SearX

Pydantic
"""

from typing import List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pathlib import Path
import os


class APISettings(BaseSettings):
    """API"""
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    title: str = Field(default="RAG-SearX API", env="API_TITLE")
    version: str = Field(default="1.0.0", env="API_VERSION")
    description: str = Field(
        default="AI", 
        env="API_DESCRIPTION"
    )
    
    class Config:
        env_prefix = "API_"


class VectorDBSettings(BaseSettings):
    """"""
    
    type: str = Field(default="milvus", env="VECTOR_DB_TYPE")
    host: str = Field(default="localhost", env="VECTOR_DB_HOST")
    port: int = Field(default=19530, env="VECTOR_DB_PORT")
    database: str = Field(default="rag_searx", env="VECTOR_DB_DATABASE")
    username: Optional[str] = Field(default=None, env="VECTOR_DB_USERNAME")
    password: Optional[str] = Field(default=None, env="VECTOR_DB_PASSWORD")
    
    # Milvus
    milvus_collection_name: str = Field(
        default="multimodal_collection", 
        env="MILVUS_COLLECTION_NAME"
    )
    milvus_index_type: str = Field(default="IVF_FLAT", env="MILVUS_INDEX_TYPE")
    milvus_metric_type: str = Field(default="L2", env="MILVUS_METRIC_TYPE")
    
    # Qdrant
    qdrant_collection_name: str = Field(
        default="multimodal_collection", 
        env="QDRANT_COLLECTION_NAME"
    )
    qdrant_vector_size: int = Field(default=768, env="QDRANT_VECTOR_SIZE")
    
    # ChromaDB
    chromadb_persist_directory: str = Field(
        default="./data/chromadb", 
        env="CHROMADB_PERSIST_DIRECTORY"
    )
    
    @validator('type')
    def validate_db_type(cls, v):
        valid_types = ['milvus', 'qdrant', 'chromadb']
        if v not in valid_types:
            raise ValueError(f': {valid_types}')
        return v
    
    class Config:
        env_prefix = "VECTOR_DB_"


class LLMSettings(BaseSettings):
    """"""
    
    provider: str = Field(default="openai", env="LLM_PROVIDER")
    model: str = Field(default="gpt-4o", env="LLM_MODEL")
    api_key: str = Field(default="", env="LLM_API_KEY")
    base_url: str = Field(
        default="https://api.openai.com/v1", 
        env="LLM_BASE_URL"
    )
    max_tokens: int = Field(default=4096, env="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    top_p: float = Field(default=0.9, env="LLM_TOP_P")
    
    # Anthropic Claude
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022", 
        env="ANTHROPIC_MODEL"
    )
    
    # Google Gemini
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-1.5-pro", env="GOOGLE_MODEL")
    
    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434", 
        env="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama2", env="OLLAMA_MODEL")
    
    @validator('provider')
    def validate_provider(cls, v):
        valid_providers = ['openai', 'anthropic', 'google', 'ollama']
        if v not in valid_providers:
            raise ValueError(f'LLM: {valid_providers}')
        return v
    
    class Config:
        env_prefix = "LLM_"


class VLMSettings(BaseSettings):
    """"""
    
    provider: str = Field(default="openai", env="VLM_PROVIDER")
    model: str = Field(default="gpt-4o", env="VLM_MODEL")
    api_key: str = Field(default="", env="VLM_API_KEY")
    base_url: str = Field(
        default="https://api.openai.com/v1", 
        env="VLM_BASE_URL"
    )
    max_tokens: int = Field(default=4096, env="VLM_MAX_TOKENS")
    
    @validator('provider')
    def validate_provider(cls, v):
        valid_providers = ['openai', 'anthropic', 'google']
        if v not in valid_providers:
            raise ValueError(f'VLM: {valid_providers}')
        return v
    
    class Config:
        env_prefix = "VLM_"


class EmbeddingSettings(BaseSettings):
    """"""
    
    provider: str = Field(default="jina", env="EMBEDDING_PROVIDER")
    model: str = Field(
        default="jina-embeddings-v2-base-zh", 
        env="EMBEDDING_MODEL"
    )
    api_key: str = Field(default="", env="EMBEDDING_API_KEY")
    dimension: int = Field(default=768, env="EMBEDDING_DIMENSION")
    batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    
    # OpenAI
    openai_embedding_model: str = Field(
        default="text-embedding-3-large", 
        env="OPENAI_EMBEDDING_MODEL"
    )
    openai_embedding_dimension: int = Field(
        default=3072, 
        env="OPENAI_EMBEDDING_DIMENSION"
    )
    
    # HuggingFace
    hf_embedding_model: str = Field(
        default="BAAI/bge-large-zh-v1.5", 
        env="HF_EMBEDDING_MODEL"
    )
    hf_api_token: str = Field(default="", env="HF_API_TOKEN")
    
    # Sentence Transformers
    st_embedding_model: str = Field(
        default="all-MiniLM-L6-v2", 
        env="ST_EMBEDDING_MODEL"
    )
    
    # 
    image_embedding_model: str = Field(
        default="clip-ViT-B-32", 
        env="IMAGE_EMBEDDING_MODEL"
    )
    image_embedding_dimension: int = Field(
        default=512, 
        env="IMAGE_EMBEDDING_DIMENSION"
    )
    
    # 
    video_embedding_model: str = Field(
        default="clip-ViT-B-32", 
        env="VIDEO_EMBEDDING_MODEL"
    )
    video_embedding_dimension: int = Field(
        default=512, 
        env="VIDEO_EMBEDDING_DIMENSION"
    )
    
    @validator('provider')
    def validate_provider(cls, v):
        valid_providers = ['jina', 'openai', 'huggingface', 'sentence_transformers']
        if v not in valid_providers:
            raise ValueError(f': {valid_providers}')
        return v
    
    class Config:
        env_prefix = "EMBEDDING_"


class RetrievalSettings(BaseSettings):
    """"""
    
    top_k_retrieval: int = Field(default=20, env="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=5, env="TOP_K_RERANK")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    hybrid_search_alpha: float = Field(default=0.7, env="HYBRID_SEARCH_ALPHA")
    enable_rerank: bool = Field(default=True, env="ENABLE_RERANK")
    rerank_model: str = Field(
        default="ms-marco-MiniLM-L-6-v2", 
        env="RERANK_MODEL"
    )
    
    @validator('similarity_threshold', 'hybrid_search_alpha')
    def validate_float_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('0.01.0')
        return v
    
    class Config:
        env_prefix = ""


class ContextSettings(BaseSettings):
    """"""
    
    max_context_length: int = Field(default=128000, env="MAX_CONTEXT_LENGTH")
    enable_context_compression: bool = Field(
        default=True, 
        env="ENABLE_CONTEXT_COMPRESSION"
    )
    compression_ratio: float = Field(default=0.5, env="COMPRESSION_RATIO")
    compression_model: str = Field(
        default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        env="COMPRESSION_MODEL"
    )
    
    @validator('compression_ratio')
    def validate_compression_ratio(cls, v):
        if not 0.1 <= v <= 1.0:
            raise ValueError('0.11.0')
        return v
    
    class Config:
        env_prefix = ""


class DocumentSettings(BaseSettings):
    """"""
    
    text_chunk_size: int = Field(default=1000, env="TEXT_CHUNK_SIZE")
    text_chunk_overlap: int = Field(default=200, env="TEXT_CHUNK_OVERLAP")
    summary_max_length: int = Field(default=500, env="SUMMARY_MAX_LENGTH")
    enable_semantic_chunking: bool = Field(
        default=True, 
        env="ENABLE_SEMANTIC_CHUNKING"
    )
    
    class Config:
        env_prefix = ""


class MediaSettings(BaseSettings):
    """"""
    
    # 
    image_caption_max_length: int = Field(
        default=200, 
        env="IMAGE_CAPTION_MAX_LENGTH"
    )
    image_compression_quality: int = Field(
        default=85, 
        env="IMAGE_COMPRESSION_QUALITY"
    )
    image_max_size: int = Field(default=1024, env="IMAGE_MAX_SIZE")
    supported_image_formats: str = Field(
        default="jpg,jpeg,png,bmp,gif,webp", 
        env="SUPPORTED_IMAGE_FORMATS"
    )
    
    # 
    video_clip_duration: int = Field(default=30, env="VIDEO_CLIP_DURATION")
    keyframe_extract_interval: int = Field(
        default=5, 
        env="KEYFRAME_EXTRACT_INTERVAL"
    )
    audio_transcription_language: str = Field(
        default="zh", 
        env="AUDIO_TRANSCRIPTION_LANGUAGE"
    )
    supported_video_formats: str = Field(
        default="mp4,avi,mov,mkv,wmv", 
        env="SUPPORTED_VIDEO_FORMATS"
    )
    
    @property
    def image_formats_list(self) -> List[str]:
        return [fmt.strip() for fmt in self.supported_image_formats.split(',')]
    
    @property
    def video_formats_list(self) -> List[str]:
        return [fmt.strip() for fmt in self.supported_video_formats.split(',')]
    
    class Config:
        env_prefix = ""


class CacheSettings(BaseSettings):
    """"""
    
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    cache_type: str = Field(default="redis", env="CACHE_TYPE")
    
    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_expire_time: int = Field(default=3600, env="REDIS_EXPIRE_TIME")
    
    @validator('cache_type')
    def validate_cache_type(cls, v):
        valid_types = ['redis', 'memory']
        if v not in valid_types:
            raise ValueError(f': {valid_types}')
        return v
    
    class Config:
        env_prefix = ""


class DatabaseSettings(BaseSettings):
    """"""
    
    database_url: str = Field(
        default="sqlite:///./data/rag_searx.db", 
        env="DATABASE_URL"
    )
    
    class Config:
        env_prefix = ""


class StorageSettings(BaseSettings):
    """"""
    
    file_storage_root: str = Field(default="./data", env="FILE_STORAGE_ROOT")
    max_upload_size: int = Field(default=100, env="MAX_UPLOAD_SIZE")  # MB
    media_url_prefix: str = Field(
        default="http://localhost:8000/media", 
        env="MEDIA_URL_PREFIX"
    )
    
    @property
    def storage_path(self) -> Path:
        return Path(self.file_storage_root)
    
    class Config:
        env_prefix = ""


class LoggingSettings(BaseSettings):
    """"""
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/rag_searx.log", env="LOG_FILE")
    log_max_size: int = Field(default=50, env="LOG_MAX_SIZE")  # MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f': {valid_levels}')
        return v.upper()
    
    class Config:
        env_prefix = ""


class SecuritySettings(BaseSettings):
    """"""
    
    jwt_secret_key: str = Field(
        default="your_jwt_secret_key_change_this_in_production", 
        env="JWT_SECRET_KEY"
    )
    jwt_expire_hours: int = Field(default=24, env="JWT_EXPIRE_HOURS")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080", 
        env="CORS_ORIGINS"
    )
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(',')]
    
    class Config:
        env_prefix = ""


class DevelopmentSettings(BaseSettings):
    """"""
    
    debug: bool = Field(default=False, env="DEBUG")
    enable_docs: bool = Field(default=True, env="ENABLE_DOCS")
    enable_verbose_errors: bool = Field(
        default=False, 
        env="ENABLE_VERBOSE_ERRORS"
    )
    
    class Config:
        env_prefix = ""


class PerformanceSettings(BaseSettings):
    """"""
    
    max_concurrent_requests: int = Field(
        default=10, 
        env="MAX_CONCURRENT_REQUESTS"
    )
    db_pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    vector_search_batch_size: int = Field(
        default=100, 
        env="VECTOR_SEARCH_BATCH_SIZE"
    )
    
    class Config:
        env_prefix = ""


class ExperimentalSettings(BaseSettings):
    """"""
    
    enable_multimodal_retrieval: bool = Field(
        default=True, 
        env="ENABLE_MULTIMODAL_RETRIEVAL"
    )
    enable_smart_rerank: bool = Field(
        default=True, 
        env="ENABLE_SMART_RERANK"
    )
    enable_adaptive_chunking: bool = Field(
        default=True, 
        env="ENABLE_ADAPTIVE_CHUNKING"
    )
    
    class Config:
        env_prefix = ""


class Settings(BaseSettings):
    """ - """
    
    api: APISettings = APISettings()
    vector_db: VectorDBSettings = VectorDBSettings()
    llm: LLMSettings = LLMSettings()
    vlm: VLMSettings = VLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    context: ContextSettings = ContextSettings()
    document: DocumentSettings = DocumentSettings()
    media: MediaSettings = MediaSettings()
    cache: CacheSettings = CacheSettings()
    database: DatabaseSettings = DatabaseSettings()
    storage: StorageSettings = StorageSettings()
    logging: LoggingSettings = LoggingSettings()
    security: SecuritySettings = SecuritySettings()
    development: DevelopmentSettings = DevelopmentSettings()
    performance: PerformanceSettings = PerformanceSettings()
    experimental: ExperimentalSettings = ExperimentalSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 
settings = Settings()


def get_settings() -> Settings:
    """"""
    return settings


def reload_settings() -> Settings:
    """"""
    global settings
    settings = Settings()
    return settings 