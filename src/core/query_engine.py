"""


RAG
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config.settings import get_settings
from ..utils.logger import get_logger
from ..api.schemas import (
    QueryRequest, QueryResponse, QueryConfig,
    RetrievedMedia, RetrievedTextCitation, DebugInfo,
    GenerationContext, RetrievalResult, ChunkData
)
from ..vectorstore.milvus_store import get_milvus_store
from ..generator.dashscope_llm import get_dashscope_llm
from ..generator.embedding_models import get_embedding_model
from ..retriever.hybrid_retriever import HybridRetriever
from ..generator.context_compressor import ContextCompressor

# 
settings = get_settings()
logger = get_logger(__name__)


@dataclass
class QueryStats:
    """"""
    total_time_ms: int
    retrieval_time_ms: int
    rerank_time_ms: int
    generation_time_ms: int
    retrieved_count: int
    reranked_count: int
    context_compressed: bool
    token_usage: Dict[str, int]


class QueryEngine:
    """"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.vector_store = None
        self.llm = None
        self.embedding_model = None
        self.hybrid_retriever = None
        self.context_compressor = None
        
    async def initialize(self):
        """"""
        try:
            self.logger.info("...")
            
            # 
            self.vector_store = await get_milvus_store()
            
            # LLM
            self.llm = get_dashscope_llm()
            
            # 
            self.embedding_model = get_embedding_model()
            
            # 
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model
            )
            
            # 
            self.context_compressor = ContextCompressor()
            
            self.logger.info("")
            
        except Exception as e:
            self.logger.error(f": {e}")
            raise
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """"""
        
        start_time = time.time()
        query_stats = QueryStats(
            total_time_ms=0,
            retrieval_time_ms=0,
            rerank_time_ms=0,
            generation_time_ms=0,
            retrieved_count=0,
            reranked_count=0,
            context_compressed=False,
            token_usage={}
        )
        
        try:
            self.logger.info(f": {request.query_text[:50]}...")
            
            # 1. 
            retrieval_start = time.time()
            retrieved_results = await self._hybrid_retrieval(request)
            query_stats.retrieval_time_ms = int((time.time() - retrieval_start) * 1000)
            query_stats.retrieved_count = len(retrieved_results)
            
            # 2. 
            rerank_start = time.time()
            if request.config.enable_rerank and retrieved_results:
                retrieved_results = await self._rerank_results(request.query_text, retrieved_results)
            query_stats.rerank_time_ms = int((time.time() - rerank_start) * 1000)
            query_stats.reranked_count = len(retrieved_results)
            
            # 3. 
            context = await self._manage_context(request, retrieved_results)
            query_stats.context_compressed = context.compressed
            
            # 4. 
            generation_start = time.time()
            answer_text = await self._generate_answer(request.query_text, context)
            query_stats.generation_time_ms = int((time.time() - generation_start) * 1000)
            
            # 5. 
            response = await self._build_response(
                request, 
                answer_text, 
                retrieved_results, 
                query_stats
            )
            
            query_stats.total_time_ms = int((time.time() - start_time) * 1000)
            response.debug_info.query_time_ms = query_stats.total_time_ms
            
            self.logger.info(f": {query_stats.total_time_ms}ms")
            return response
            
        except Exception as e:
            self.logger.error(f": {e}")
            # 
            return QueryResponse(
                answer_text=f"{str(e)}",
                retrieved_media=[],
                retrieved_text_citations=[],
                debug_info=DebugInfo(
                    query_time_ms=int((time.time() - start_time) * 1000),
                    llm_model_used=self.settings.llm.model,
                    retrieval_stats={"error": str(e)}
                )
            )
    
    async def _hybrid_retrieval(self, request: QueryRequest) -> List[RetrievalResult]:
        """"""
        
        try:
            # 
            query_vector = await self.embedding_model.encode_text(request.query_text)
            
            # 
            search_results = await self.hybrid_retriever.hybrid_search(
                query_text=request.query_text,
                query_vector=query_vector,
                top_k=request.config.top_k_retrieval,
                enable_multimedia=request.config.enable_multimedia
            )
            
            # RetrievalResult
            retrieval_results = []
            for i, result in enumerate(search_results):
                chunk_data = ChunkData(
                    chunk_id=result.id,
                    content=result.content,
                    chunk_type=result.metadata.media_type,
                    metadata=result.metadata,
                    embeddings=result.embedding
                )
                
                retrieval_result = RetrievalResult(
                    chunk_data=chunk_data,
                    score=result.score,
                    rank=i + 1
                )
                retrieval_results.append(retrieval_result)
            
            self.logger.info(f" {len(retrieval_results)} ")
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f": {e}")
            return []
    
    async def _rerank_results(
        self, 
        query: str, 
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """"""
        
        try:
            # 
            # top-k
            top_k = self.settings.retrieval.top_k_rerank
            
            # k
            sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
            reranked_results = sorted_results[:top_k]
            
            # 
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
            
            self.logger.info(f" {len(reranked_results)} ")
            return reranked_results
            
        except Exception as e:
            self.logger.error(f": {e}")
            return results
    
    async def _manage_context(
        self, 
        request: QueryRequest, 
        results: List[RetrievalResult]
    ) -> GenerationContext:
        """"""
        
        try:
            # token
            total_tokens = sum(
                len(result.chunk_data.content.split()) * 1.3  # token
                for result in results
            )
            
            context = GenerationContext(
                query=request.query_text,
                retrieved_chunks=results,
                total_tokens=int(total_tokens),
                compressed=False
            )
            
            # 
            if (request.config.enable_context_compression and 
                total_tokens > self.settings.context.max_context_length * 0.7):
                
                try:
                    # 
                    compressed_context = await self.context_compressor.compress_context(context)
                    self.logger.info(f" {total_tokens}  {compressed_context.total_tokens}")
                    return compressed_context
                except Exception as e:
                    self.logger.warning(f": {e}")
            
            return context
            
        except Exception as e:
            self.logger.error(f": {e}")
            # 
            return GenerationContext(
                query=request.query_text,
                retrieved_chunks=[],
                total_tokens=0,
                compressed=False
            )
    
    async def _generate_answer(self, query: str, context: GenerationContext) -> str:
        """"""
        
        try:
            if not context.retrieved_chunks:
                return ""
            
            # LLM
            answer = await self.llm.generate_answer_with_context(query, context)
            
            return answer
            
        except Exception as e:
            self.logger.error(f": {e}")
            return f"{str(e)}"
    
    async def _build_response(
        self,
        request: QueryRequest,
        answer_text: str,
        results: List[RetrievalResult],
        stats: QueryStats
    ) -> QueryResponse:
        """"""
        
        try:
            # 
            text_results = []
            media_results = []
            
            for result in results:
                chunk_data = result.chunk_data
                citation = f"{chunk_data.metadata.context_link.chapter_number} {chunk_data.metadata.context_link.chapter_title}{chunk_data.metadata.context_link.page_number}"
                
                if chunk_data.chunk_type.value in ["text_summary", "text_detail"]:
                    text_citation = RetrievedTextCitation(
                        text_snippet=chunk_data.content[:200] + "..." if len(chunk_data.content) > 200 else chunk_data.content,
                        source_citation=citation,
                        relevance_score=result.score
                    )
                    text_results.append(text_citation)
                
                elif chunk_data.chunk_type.value in ["image", "video_clip"] and request.config.enable_multimedia:
                    media_type = "image" if chunk_data.chunk_type.value == "image" else "video_clip"
                    media_item = RetrievedMedia(
                        type=media_type,
                        url=chunk_data.metadata.source_url or f"/media/{chunk_data.chunk_id}",
                        description=chunk_data.content,
                        relevance_score=result.score,
                        source_citation=f": {citation}"
                    )
                    media_results.append(media_item)
            
            # 
            debug_info = DebugInfo(
                query_time_ms=stats.total_time_ms,
                llm_model_used=self.settings.llm.model,
                retrieval_stats={
                    "total_retrieved": stats.retrieved_count,
                    "after_rerank": stats.reranked_count,
                    "retrieval_time_ms": stats.retrieval_time_ms,
                    "rerank_time_ms": stats.rerank_time_ms,
                    "compression_applied": stats.context_compressed
                },
                generation_stats={
                    "generation_time_ms": stats.generation_time_ms,
                    "model_used": self.settings.llm.model
                },
                token_usage=stats.token_usage
            )
            
            return QueryResponse(
                answer_text=answer_text,
                retrieved_media=media_results[:5],  # 
                retrieved_text_citations=text_results[:3],  # 
                debug_info=debug_info
            )
            
        except Exception as e:
            self.logger.error(f": {e}")
            return QueryResponse(
                answer_text=answer_text,
                retrieved_media=[],
                retrieved_text_citations=[],
                debug_info=DebugInfo(
                    query_time_ms=stats.total_time_ms,
                    llm_model_used=self.settings.llm.model,
                    retrieval_stats={"error": str(e)}
                )
            )


# 
_query_engine: Optional[QueryEngine] = None


async def get_query_engine() -> QueryEngine:
    """"""
    global _query_engine
    
    if _query_engine is None:
        _query_engine = QueryEngine()
        await _query_engine.initialize()
    
    return _query_engine


async def process_query(request: QueryRequest) -> QueryResponse:
    """"""
    engine = await get_query_engine()
    return await engine.process_query(request) 