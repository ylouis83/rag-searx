#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow + BGE 
RAGFlowBGE-large-zh-v1.5RAG
"""

from .document_processor import (
    RAGFlowDocumentProcessor, 
    ChunkConfig, 
    ChunkingStrategy, 
    DocumentChunk
)

from .bge_embedding import (
    BGEEmbeddingModel,
    get_bge_model
)

from .ragflow_rag_pipeline import RAGFlowPipeline

__version__ = "1.0.0"
__author__ = "AI-Coding R&D Team"

__all__ = [
    "RAGFlowDocumentProcessor",
    "ChunkConfig", 
    "ChunkingStrategy",
    "DocumentChunk",
    "BGEEmbeddingModel",
    "get_bge_model",
    "RAGFlowPipeline"
] 