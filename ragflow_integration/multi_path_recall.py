#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

import re
import time
import jieba
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum

# rank_bm25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("  rank_bm25 BM25")


class RecallPath(Enum):
    """"""
    VECTOR = "vector"          # 
    KEYWORD = "keyword"        #   
    SEMANTIC = "semantic"      # 
    HYBRID = "hybrid"          # 


@dataclass
class RecallResult:
    """"""
    doc_id: str
    content: str
    score: float
    recall_path: RecallPath
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "recall_path": self.recall_path.value,
            "metadata": self.metadata
        }


@dataclass
class MultiRecallConfig:
    """"""
    # 
    vector_weight: float = 0.5
    keyword_weight: float = 0.3
    semantic_weight: float = 0.2
    
    # 
    vector_top_k: int = 20
    keyword_top_k: int = 15
    semantic_top_k: int = 10
    
    # 
    final_top_k: int = 10
    
    # 
    enable_query_expansion: bool = True
    enable_synonym_expansion: bool = True
    
    # 
    enable_custom_dict: bool = True
    
    # 
    enable_reranking: bool = True
    rerank_top_k: int = 50


class SimpleBM25:
    """BM25rank_bm25"""
    
    def __init__(self, corpus: List[List[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        
        # 
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0
        
        # 
        self.doc_freqs = []
        self.idf = {}
        
        # 
        df = defaultdict(int)
        for doc in corpus:
            words_in_doc = set(doc)
            for word in words_in_doc:
                df[word] += 1
        
        # IDF
        for word, freq in df.items():
            self.idf[word] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5))
        
        # TF
        for doc in corpus:
            doc_freqs = Counter(doc)
            self.doc_freqs.append(doc_freqs)
    
    def get_scores(self, query: List[str]) -> List[float]:
        """BM25"""
        scores = []
        
        for i, doc in enumerate(self.corpus):
            score = 0
            doc_freqs = self.doc_freqs[i]
            doc_len = self.doc_lengths[i]
            
            for word in query:
                if word in doc_freqs and word in self.idf:
                    tf = doc_freqs[word]
                    idf = self.idf[word]
                    
                    # BM25
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return scores


class MultiPathRecallSystem:
    """"""
    
    def __init__(self, config: MultiRecallConfig = None):
        """
        
        
        Args:
            config: 
        """
        self.config = config or MultiRecallConfig()
        
        # 
        self.embedding_model = None
        self.bm25_model = None
        self.documents = []
        self.doc_vectors = None
        self.doc_metadata = []
        
        # 
        self._init_tokenizer()
        
        # 
        self.synonym_dict = self._load_synonym_dict()
        
        print(" ")
        print(f"   : {self.config.vector_weight}")
        print(f"   : {self.config.keyword_weight}")
        print(f"   : {self.config.semantic_weight}")
    
    def _init_tokenizer(self):
        """"""
        # jieba
        jieba.setLogLevel('WARN')  # 
        
        # 
        if self.config.enable_custom_dict:
            custom_words = [
                "", "", "", "", "2", "1",
                "", "", "", "", "", "",
                "", "", "", "", "", "",
                "", "", "", "", "", "", ""
            ]
            
            for word in custom_words:
                jieba.add_word(word)
        
        print(" ")
    
    def _load_synonym_dict(self) -> Dict[str, List[str]]:
        """"""
        # 
        synonyms = {
            "": ["", "", ""],
            "": [""],
            "": ["", "", ""],
            "": ["", ""],
            "": ["", "", "", ""],
            "": ["", "", ""],
            "": ["", "", ""],
            "": ["", "", ""]
        }
        
        return synonyms
    
    def set_embedding_model(self, model):
        """"""
        self.embedding_model = model
        print(" ")
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """
        
        
        Args:
            documents: id, content, metadata
        """
        print(f" : {len(documents)}")
        start_time = time.time()
        
        self.documents = documents
        self.doc_metadata = [doc.get("metadata", {}) for doc in documents]
        
        # 1. 
        self._build_keyword_index()
        
        # 2. 
        if self.embedding_model:
            self._build_vector_index()
        
        build_time = time.time() - start_time
        print(f" : {build_time:.2f}")
    
    def _build_keyword_index(self):
        """BM25"""
        print(" ...")
        
        # 
        tokenized_docs = []
        for doc in self.documents:
            content = doc.get("content", "")
            tokens = self._tokenize_text(content)
            tokenized_docs.append(tokens)
        
        # BM25
        if HAS_BM25:
            self.bm25_model = BM25Okapi(tokenized_docs)
        else:
            self.bm25_model = SimpleBM25(tokenized_docs)
        
        print(f" : {len(tokenized_docs)}")
    
    def _build_vector_index(self):
        """"""
        print(" ...")
        
        # 
        doc_contents = [doc.get("content", "") for doc in self.documents]
        
        # 
        self.doc_vectors = self.embedding_model.batch_encode(doc_contents)
        
        print(f" : {self.doc_vectors.shape}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """"""
        # 
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        # jieba
        tokens = list(jieba.cut(text))
        
        # 
        filtered_tokens = [
            token.strip() for token in tokens
            if len(token.strip()) >= 2 and not self._is_stopword(token.strip())
        ]
        
        return filtered_tokens
    
    def _is_stopword(self, word: str) -> bool:
        """"""
        stopwords = {
            "", "", "", "", "", "", "", "", "", "", "", "",
            "", "", "", "", "", "", "", "", "", "", "", "",
            "", "", "", "", "", "", "", "", ""
        }
        return word in stopwords
    
    def multi_recall(self, query: str, top_k: int = None) -> List[RecallResult]:
        """
        
        
        Args:
            query: 
            top_k: 
        
        Returns:
            
        """
        top_k = top_k or self.config.final_top_k
        
        print(f" : {query}")
        start_time = time.time()
        
        # 
        expanded_queries = self._expand_query(query)
        
        # 
        all_results = []
        
        # 1. 
        if self.embedding_model and self.doc_vectors is not None:
            vector_results = self._vector_recall(query, self.config.vector_top_k)
            all_results.extend(vector_results)
            print(f"    : {len(vector_results)} ")
        
        # 2. 
        if self.bm25_model:
            keyword_results = self._keyword_recall(query, self.config.keyword_top_k)
            all_results.extend(keyword_results)
            print(f"    : {len(keyword_results)} ")
        
        # 3. 
        if self.config.enable_query_expansion and expanded_queries:
            semantic_results = self._semantic_recall(expanded_queries, self.config.semantic_top_k)
            all_results.extend(semantic_results)
            print(f"    : {len(semantic_results)} ")
        
        # 4. 
        final_results = self._fuse_and_rerank(all_results, top_k)
        
        recall_time = time.time() - start_time
        print(f" : {recall_time:.3f}: {len(final_results)} ")
        
        return final_results
    
    def _expand_query(self, query: str) -> List[str]:
        """"""
        expanded = [query]  # 
        
        if self.config.enable_synonym_expansion:
            # 
            query_tokens = self._tokenize_text(query)
            
            for token in query_tokens:
                if token in self.synonym_dict:
                    synonyms = self.synonym_dict[token]
                    for synonym in synonyms:
                        expanded_query = query.replace(token, synonym)
                        if expanded_query != query:
                            expanded.append(expanded_query)
        
        return expanded[:5]  # 5
    
    def _vector_recall(self, query: str, top_k: int) -> List[RecallResult]:
        """"""
        if not self.embedding_model or self.doc_vectors is None:
            return []
        
        # 
        query_vector = self.embedding_model.encode_query(query)
        
        # 
        similarities = self.embedding_model.compute_similarity(
            query_vector.reshape(1, -1), 
            self.doc_vectors
        ).flatten()
        
        # top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 
                results.append(RecallResult(
                    doc_id=self.documents[idx].get("id", str(idx)),
                    content=self.documents[idx].get("content", ""),
                    score=float(similarities[idx]),
                    recall_path=RecallPath.VECTOR,
                    metadata=self.doc_metadata[idx]
                ))
        
        return results
    
    def _keyword_recall(self, query: str, top_k: int) -> List[RecallResult]:
        """BM25"""
        if not self.bm25_model:
            return []
        
        # 
        query_tokens = self._tokenize_text(query)
        if not query_tokens:
            return []
        
        # BM25
        scores = self.bm25_model.get_scores(query_tokens)
        
        # top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 
                results.append(RecallResult(
                    doc_id=self.documents[idx].get("id", str(idx)),
                    content=self.documents[idx].get("content", ""),
                    score=float(scores[idx]),
                    recall_path=RecallPath.KEYWORD,
                    metadata=self.doc_metadata[idx]
                ))
        
        return results
    
    def _semantic_recall(self, expanded_queries: List[str], top_k: int) -> List[RecallResult]:
        """"""
        if not self.embedding_model or not expanded_queries:
            return []
        
        all_results = []
        
        for expanded_query in expanded_queries[1:]:  # 
            # 
            query_vector = self.embedding_model.encode_query(expanded_query)
            similarities = self.embedding_model.compute_similarity(
                query_vector.reshape(1, -1), 
                self.doc_vectors
            ).flatten()
            
            # top
            top_indices = np.argsort(similarities)[::-1][:top_k//len(expanded_queries)]
            
            for idx in top_indices:
                if similarities[idx] > 0:
                    all_results.append(RecallResult(
                        doc_id=self.documents[idx].get("id", str(idx)),
                        content=self.documents[idx].get("content", ""),
                        score=float(similarities[idx]) * 0.8,  # 
                        recall_path=RecallPath.SEMANTIC,
                        metadata=self.doc_metadata[idx]
                    ))
        
        # 
        seen_ids = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.score, reverse=True):
            if result.doc_id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.doc_id)
        
        return unique_results[:top_k]
    
    def _fuse_and_rerank(self, all_results: List[RecallResult], top_k: int) -> List[RecallResult]:
        """"""
        if not all_results:
            return []
        
        # ID
        doc_groups = defaultdict(list)
        for result in all_results:
            doc_groups[result.doc_id].append(result)
        
        # 
        fused_results = []
        for doc_id, results in doc_groups.items():
            # 
            path_scores = defaultdict(float)
            best_result = results[0]  # 
            
            for result in results:
                if result.recall_path == RecallPath.VECTOR:
                    path_scores[RecallPath.VECTOR] = max(path_scores[RecallPath.VECTOR], result.score)
                elif result.recall_path == RecallPath.KEYWORD:
                    path_scores[RecallPath.KEYWORD] = max(path_scores[RecallPath.KEYWORD], result.score)
                elif result.recall_path == RecallPath.SEMANTIC:
                    path_scores[RecallPath.SEMANTIC] = max(path_scores[RecallPath.SEMANTIC], result.score)
            
            # RRF - Reciprocal Rank Fusion
            final_score = (
                path_scores[RecallPath.VECTOR] * self.config.vector_weight +
                path_scores[RecallPath.KEYWORD] * self.config.keyword_weight +
                path_scores[RecallPath.SEMANTIC] * self.config.semantic_weight
            )
            
            # 
            path_bonus = 1.0 + 0.1 * (len(path_scores) - 1)
            final_score *= path_bonus
            
            fused_results.append(RecallResult(
                doc_id=doc_id,
                content=best_result.content,
                score=final_score,
                recall_path=RecallPath.HYBRID,
                metadata={
                    **best_result.metadata,
                    "recall_paths": list(path_scores.keys()),
                    "path_scores": dict(path_scores),
                    "path_count": len(path_scores)
                }
            ))
        
        # top_k
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        return fused_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """"""
        return {
            "total_documents": len(self.documents),
            "has_vector_index": self.doc_vectors is not None,
            "has_keyword_index": self.bm25_model is not None,
            "embedding_dim": self.doc_vectors.shape[1] if self.doc_vectors is not None else 0,
            "config": {
                "vector_weight": self.config.vector_weight,
                "keyword_weight": self.config.keyword_weight,
                "semantic_weight": self.config.semantic_weight,
                "final_top_k": self.config.final_top_k
            }
        }


# 
if __name__ == "__main__":
    print(" ")
    print("=" * 50)
    
    # 
    test_documents = [
        {
            "id": "doc_1",
            "content": "2",
            "metadata": {"type": "medical", "category": "diabetes"}
        },
        {
            "id": "doc_2", 
            "content": "",
            "metadata": {"type": "medical", "category": "hormone"}
        },
        {
            "id": "doc_3",
            "content": "",
            "metadata": {"type": "lifestyle", "category": "prevention"}
        }
    ]
    
    # 
    config = MultiRecallConfig(
        vector_weight=0.4,
        keyword_weight=0.4,
        semantic_weight=0.2,
        final_top_k=5
    )
    
    # 
    recall_system = MultiPathRecallSystem(config)
    
    # 
    recall_system.build_index(test_documents)
    
    # 
    test_query = ""
    print(f"\n : {test_query}")
    
    # embedding_model
    results = recall_system.multi_recall(test_query, top_k=3)
    
    print(f"\n :")
    for i, result in enumerate(results, 1):
        print(f"   {i}. ID: {result.doc_id}")
        print(f"      : {result.score:.4f}")
        print(f"      : {result.recall_path.value}")
        print(f"      : {result.content[:100]}...")
        print()
    
    # 
    stats = recall_system.get_stats()
    print(" :")
    for key, value in stats.items():
        print(f"   {key}: {value}") 