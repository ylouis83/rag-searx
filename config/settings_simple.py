"""
RAG-SearX
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """"""
    
    # API
    api_host: str = Field(default="0.0.0.0", env="HOST")
    api_port: int = Field(default=8000, env="PORT")
    api_title: str = Field(default="RAG-SearX API", env="APP_NAME")
    api_version: str = Field(default="1.0.0", env="APP_VERSION")
    api_description: str = Field(default="AI")
    
    # 
    debug: bool = Field(default=True, env="DEBUG")
    enable_docs: bool = Field(default=True)
    enable_verbose_errors: bool = Field(default=True)
    
    # LLM
    llm_provider: str = Field(default="dashscope", env="LLM_PROVIDER")
    llm_api_key: str = Field(default="", env="LLM_API_KEY")
    llm_model: str = Field(default="qwen-plus", env="LLM_MODEL")
    llm_max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_top_p: float = Field(default=0.9, env="LLM_TOP_P")
    
    # 
    embedding_provider: str = Field(default="dashscope", env="EMBEDDING_PROVIDER")
    embedding_api_key: str = Field(default="", env="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="text-embedding-v1", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    
    # 
    vector_db_type: str = Field(default="milvus", env="VECTOR_DB_PROVIDER")
    vector_db_host: str = Field(default="localhost", env="VECTOR_DB_HOST")
    vector_db_port: int = Field(default=19530, env="VECTOR_DB_PORT")
    milvus_collection_name: str = Field(default="rag_searx_collection", env="MILVUS_COLLECTION_NAME")
    
    # 
    retrieval_top_k_retrieval: int = Field(default=20, env="RETRIEVAL_TOP_K_RETRIEVAL")
    retrieval_top_k_rerank: int = Field(default=10, env="RETRIEVAL_TOP_K_RERANK")
    
    # 
    cors_origins: str = Field(default="http://localhost:3000,http://127.0.0.1:3000", env="ALLOWED_ORIGINS")
    
    # 
    storage_path: str = Field(default="./data", env="UPLOAD_DIR")
    
    @property
    def cors_origins_list(self) -> List[str]:
        """CORS"""
        return [origin.strip() for origin in self.cors_origins.split(',')]
    
    model_config = {"env_file": ".env", "extra": "allow"}


# 
settings = Settings()


def get_settings() -> Settings:
    """"""
    return settings 