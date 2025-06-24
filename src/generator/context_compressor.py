"""


LLMToken
"""

import re
from typing import List, Dict, Any, Optional
import jieba
from collections import Counter

from config.settings import get_settings
from ..utils.logger import get_logger
from ..api.schemas import GenerationContext, RetrievalResult

# 
settings = get_settings()
logger = get_logger(__name__)


class ContextCompressor:
    """"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.max_context_length = self.settings.context.max_context_length
        self.compression_ratio = self.settings.context.compression_ratio
        
        # 
        jieba.initialize()
        
    async def compress_context(self, context: GenerationContext) -> GenerationContext:
        """"""
        
        try:
            self.logger.info(f"Token: {context.total_tokens}")
            
            if context.total_tokens <= self.max_context_length:
                self.logger.info("")
                return context
            
            # 1. 
            sorted_chunks = await self._rank_chunks_by_importance(
                context.retrieved_chunks, 
                context.query
            )
            
            # 2. 
            selected_chunks = await self._select_chunks(
                sorted_chunks, 
                target_ratio=self.compression_ratio
            )
            
            # 3. 
            compressed_chunks = []
            for chunk in selected_chunks:
                compressed_chunk = await self._compress_chunk(chunk, context.query)
                compressed_chunks.append(compressed_chunk)
            
            # 4. Token
            new_total_tokens = sum(
                len(chunk.chunk_data.content.split()) * 1.3  # 
                for chunk in compressed_chunks
            )
            
            compressed_context = GenerationContext(
                query=context.query,
                retrieved_chunks=compressed_chunks,
                total_tokens=int(new_total_tokens),
                compressed=True,
                compression_ratio=len(compressed_chunks) / len(context.retrieved_chunks) if context.retrieved_chunks else 0
            )
            
            self.logger.info(
                f" {len(context.retrieved_chunks)}  {len(compressed_chunks)} "
                f"Token {context.total_tokens}  {new_total_tokens}"
            )
            
            return compressed_context
            
        except Exception as e:
            self.logger.error(f": {e}")
            # 
            return await self._fallback_truncate(context)
    
    async def _rank_chunks_by_importance(
        self, 
        chunks: List[RetrievalResult], 
        query: str
    ) -> List[RetrievalResult]:
        """"""
        
        try:
            query_keywords = self._extract_keywords(query)
            
            # 
            for chunk in chunks:
                importance_score = await self._calculate_importance_score(
                    chunk, 
                    query_keywords
                )
                # 
                chunk.score = chunk.score * 0.7 + importance_score * 0.3
            
            # 
            sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
            
            return sorted_chunks
            
        except Exception as e:
            self.logger.error(f": {e}")
            return chunks
    
    async def _calculate_importance_score(
        self, 
        chunk: RetrievalResult, 
        query_keywords: List[str]
    ) -> float:
        """"""
        
        try:
            content = chunk.chunk_data.content
            score = 0.0
            
            # 1. 
            keyword_score = self._calculate_keyword_match_score(content, query_keywords)
            score += keyword_score * 0.4
            
            # 2. 
            media_type = chunk.chunk_data.chunk_type.value
            if media_type == "text_summary":
                score += 0.3  # 
            elif media_type == "text_detail":
                score += 0.2
            elif media_type in ["image", "video_clip"]:
                score += 0.1
            
            # 3. 
            content_length = len(content)
            if 100 <= content_length <= 500:
                score += 0.2
            elif 50 <= content_length <= 1000:
                score += 0.1
            
            # 4. 
            position_score = max(0, 1.0 - (chunk.rank - 1) * 0.1)
            score += position_score * 0.1
            
            return min(score, 1.0)  # [0, 1]
            
        except Exception as e:
            self.logger.error(f": {e}")
            return 0.5  # 
    
    def _calculate_keyword_match_score(self, content: str, keywords: List[str]) -> float:
        """"""
        
        try:
            if not keywords:
                return 0.0
            
            content_lower = content.lower()
            matched_keywords = 0
            total_matches = 0
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                count = content_lower.count(keyword_lower)
                if count > 0:
                    matched_keywords += 1
                    total_matches += count
            
            # 
            match_rate = matched_keywords / len(keywords)
            match_density = total_matches / len(content.split())
            
            return (match_rate * 0.7 + match_density * 0.3)
            
        except Exception as e:
            self.logger.error(f": {e}")
            return 0.0
    
    async def _select_chunks(
        self, 
        sorted_chunks: List[RetrievalResult], 
        target_ratio: float
    ) -> List[RetrievalResult]:
        """"""
        
        try:
            target_count = max(1, int(len(sorted_chunks) * target_ratio))
            
            # 
            text_chunks = [c for c in sorted_chunks if c.chunk_data.chunk_type.value in ["text_summary", "text_detail"]]
            media_chunks = [c for c in sorted_chunks if c.chunk_data.chunk_type.value in ["image", "video_clip"]]
            
            selected_chunks = []
            
            # 
            text_count = min(target_count - 1, len(text_chunks))
            selected_chunks.extend(text_chunks[:text_count])
            
            # 
            remaining_count = target_count - len(selected_chunks)
            if remaining_count > 0 and media_chunks:
                media_count = min(remaining_count, len(media_chunks))
                selected_chunks.extend(media_chunks[:media_count])
            
            # 
            remaining_count = target_count - len(selected_chunks)
            if remaining_count > 0 and len(text_chunks) > text_count:
                additional_text = text_chunks[text_count:text_count + remaining_count]
                selected_chunks.extend(additional_text)
            
            return selected_chunks
            
        except Exception as e:
            self.logger.error(f": {e}")
            # 
            target_count = max(1, int(len(sorted_chunks) * target_ratio))
            return sorted_chunks[:target_count]
    
    async def _compress_chunk(
        self, 
        chunk: RetrievalResult, 
        query: str
    ) -> RetrievalResult:
        """"""
        
        try:
            content = chunk.chunk_data.content
            
            # 
            if len(content) <= 200:
                return chunk
            
            # 
            query_keywords = self._extract_keywords(query)
            
            # 
            compressed_content = self._compress_text(content, query_keywords)
            
            # 
            compressed_chunk = RetrievalResult(
                chunk_data=chunk.chunk_data.copy(),
                score=chunk.score,
                rank=chunk.rank
            )
            compressed_chunk.chunk_data.content = compressed_content
            
            return compressed_chunk
            
        except Exception as e:
            self.logger.error(f": {e}")
            return chunk
    
    def _compress_text(self, text: str, keywords: List[str]) -> str:
        """"""
        
        try:
            # 1. 
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 3:
                return text  # 
            
            # 2. 
            sentence_scores = []
            for sentence in sentences:
                score = self._calculate_sentence_importance(sentence, keywords)
                sentence_scores.append((sentence, score))
            
            # 3. 
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 4. 
            target_count = max(2, len(sentences) // 2)  # 
            selected_sentences = sentence_scores[:target_count]
            
            # 5. 
            selected_texts = [s[0] for s in selected_sentences]
            original_order = []
            for sentence in sentences:
                if sentence in selected_texts:
                    original_order.append(sentence)
            
            compressed_text = ''.join(original_order) + ''
            
            return compressed_text
            
        except Exception as e:
            self.logger.error(f": {e}")
            # 
            return text[:300] + "..." if len(text) > 300 else text
    
    def _calculate_sentence_importance(self, sentence: str, keywords: List[str]) -> float:
        """"""
        
        try:
            score = 0.0
            
            # 
            if keywords:
                matched_keywords = sum(
                    1 for keyword in keywords 
                    if keyword.lower() in sentence.lower()
                )
                score += matched_keywords / len(keywords) * 0.6
            
            # 
            length = len(sentence)
            if 20 <= length <= 100:
                score += 0.3
            elif 10 <= length <= 150:
                score += 0.2
            
            # 
            # 
            score += 0.1
            
            return score
            
        except Exception as e:
            self.logger.error(f": {e}")
            return 0.1
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """"""
        
        try:
            # 
            words = jieba.lcut(text, cut_all=False)
            
            # 
            stopwords = {
                '', '', '', '', '', '', '', '', '', '', '', 
                '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', ''
            }
            
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
    
    async def _fallback_truncate(self, context: GenerationContext) -> GenerationContext:
        """"""
        
        try:
            # 
            target_count = max(1, int(len(context.retrieved_chunks) * 0.5))
            truncated_chunks = context.retrieved_chunks[:target_count]
            
            new_total_tokens = sum(
                len(chunk.chunk_data.content.split()) * 1.3
                for chunk in truncated_chunks
            )
            
            return GenerationContext(
                query=context.query,
                retrieved_chunks=truncated_chunks,
                total_tokens=int(new_total_tokens),
                compressed=True,
                compression_ratio=0.5
            )
            
        except Exception as e:
            self.logger.error(f": {e}")
            return context 