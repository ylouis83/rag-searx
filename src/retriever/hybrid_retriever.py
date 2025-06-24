"""



"""

from typing import List, Optional, Dict, Any
import re
import jieba
from collections import Counter

from config.settings import get_settings
from ..utils.logger import get_logger
from ..vectorstore.base import VectorStore, SearchResult
from ..generator.embedding_models import EmbeddingModel

# 
settings = get_settings()
logger = get_logger(__name__)


class HybridRetriever:
    """"""
    
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.settings = settings
        self.logger = logger
        
        # 
        jieba.initialize()
        
    async def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 20,
        enable_multimedia: bool = True,
        alpha: float = None
    ) -> List[SearchResult]:
        """
        
        
        Args:
            query_text: 
            query_vector: 
            top_k: 
            enable_multimedia: 
            alpha:  (0.0-1.0) (1-alpha)
        """
        
        try:
            if alpha is None:
                alpha = self.settings.retrieval.hybrid_search_alpha
                
            self.logger.info(f"alpha={alpha}, top_k={top_k}")
            
            # 1. 
            vector_results = await self._vector_search(
                query_vector, 
                top_k * 2,  # 
                enable_multimedia
            )
            
            # 2. 
            keyword_results = await self._keyword_search(
                query_text, 
                top_k * 2,
                enable_multimedia
            )
            
            # 3. 
            fused_results = await self._fuse_results(
                vector_results, 
                keyword_results, 
                alpha, 
                top_k
            )
            
            self.logger.info(f" {len(fused_results)} ")
            return fused_results
            
        except Exception as e:
            self.logger.error(f": {e}")
            # 
            return await self._vector_search(query_vector, top_k, enable_multimedia)
    
    async def _vector_search(
        self, 
        query_vector: List[float], 
        top_k: int,
        enable_multimedia: bool
    ) -> List[SearchResult]:
        """"""
        
        try:
            collection_name = self.settings.vector_db.milvus_collection_name
            
            # 
            filter_expr = None
            if not enable_multimedia:
                filter_expr = "media_type in ['text_summary', 'text_detail']"
            
            # 
            results = await self.vector_store.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
                filter_expr=filter_expr
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f": {e}")
            return []
    
    async def _keyword_search(
        self, 
        query_text: str, 
        top_k: int,
        enable_multimedia: bool
    ) -> List[SearchResult]:
        """"""
        
        try:
            # 
            keywords = self._extract_keywords(query_text)
            
            if not keywords:
                return []
            
            # 
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append(f"content like '%{keyword}%'")
            
            filter_expr = " or ".join(keyword_conditions)
            
            # 
            if not enable_multimedia:
                filter_expr = f"({filter_expr}) and media_type in ['text_summary', 'text_detail']"
            
            collection_name = self.settings.vector_db.milvus_collection_name
            
            # 
            zero_vector = [0.0] * self.embedding_model.get_dimension()
            
            results = await self.vector_store.search_vectors(
                collection_name=collection_name,
                query_vector=zero_vector,
                top_k=top_k,
                filter_expr=filter_expr
            )
            
            # 
            for result in results:
                result.score = self._calculate_keyword_score(result.content, keywords)
            
            # 
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f": {e}")
            return []
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """"""
        
        try:
            # 
            words = jieba.lcut(text, cut_all=False)
            
            # 
            stopwords = {'', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''}
            
            filtered_words = [
                word.strip() for word in words 
                if len(word.strip()) > 1 and word.strip() not in stopwords
            ]
            
            # 
            word_freq = Counter(filtered_words)
            
            # 
            keywords = [word for word, freq in word_freq.most_common(max_keywords)]
            
            return keywords
            
        except Exception as e:
            self.logger.error(f": {e}")
            return []
    
    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """"""
        
        try:
            if not keywords:
                return 0.0
            
            content_lower = content.lower()
            score = 0.0
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # 
                count = content_lower.count(keyword_lower)
                if count > 0:
                    # TF
                    tf = count / len(content.split())
                    score += tf
            
            return score
            
        except Exception as e:
            self.logger.error(f": {e}")
            return 0.0
    
    async def _fuse_results(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        alpha: float,
        top_k: int
    ) -> List[SearchResult]:
        """"""
        
        try:
            # 
            result_dict: Dict[str, SearchResult] = {}
            
            # 
            if vector_results:
                max_vector_score = max(result.score for result in vector_results)
                min_vector_score = min(result.score for result in vector_results)
                score_range = max_vector_score - min_vector_score
                
                for result in vector_results:
                    if score_range > 0:
                        normalized_score = (result.score - min_vector_score) / score_range
                    else:
                        normalized_score = 1.0
                    
                    if result.id not in result_dict:
                        result_dict[result.id] = result
                        result_dict[result.id].score = alpha * normalized_score
                    else:
                        result_dict[result.id].score += alpha * normalized_score
            
            # 
            if keyword_results:
                max_keyword_score = max(result.score for result in keyword_results)
                min_keyword_score = min(result.score for result in keyword_results)
                score_range = max_keyword_score - min_keyword_score if max_keyword_score > min_keyword_score else 1.0
                
                for result in keyword_results:
                    if score_range > 0:
                        normalized_score = (result.score - min_keyword_score) / score_range
                    else:
                        normalized_score = 1.0
                    
                    if result.id not in result_dict:
                        result_dict[result.id] = result
                        result_dict[result.id].score = (1 - alpha) * normalized_score
                    else:
                        result_dict[result.id].score += (1 - alpha) * normalized_score
            
            # 
            fused_results = list(result_dict.values())
            fused_results.sort(key=lambda x: x.score, reverse=True)
            
            # top-k
            return fused_results[:top_k]
            
        except Exception as e:
            self.logger.error(f": {e}")
            # 
            return vector_results[:top_k] if vector_results else []
    
    async def semantic_search(
        self,
        query_text: str,
        top_k: int = 10,
        media_type_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """"""
        
        try:
            # 
            query_vector = await self.embedding_model.encode_text(query_text)
            
            # 
            filter_expr = None
            if media_type_filter:
                filter_expr = f"media_type == '{media_type_filter}'"
            
            collection_name = self.settings.vector_db.milvus_collection_name
            
            results = await self.vector_store.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
                filter_expr=filter_expr
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f": {e}")
            return []
    
    async def keyword_only_search(
        self,
        query_text: str,
        top_k: int = 10,
        media_type_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """"""
        
        try:
            return await self._keyword_search(
                query_text, 
                top_k, 
                enable_multimedia=media_type_filter is None
            )
            
        except Exception as e:
            self.logger.error(f": {e}")
            return [] 