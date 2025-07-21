#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拼音向量搜索模块
解决语音识别中发音差异导致的搜索问题
例如："周萍" vs "邹萍" 应该能够统一检索
"""

import os
import re
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import difflib

# 拼音处理
from pypinyin import pinyin, lazy_pinyin, Style, load_phrases_dict

# BGE嵌入模型
from .bge_embedding import BGEEmbeddingModel, get_bge_model


@dataclass
class PinyinSearchResult:
    """拼音搜索结果"""
    entity_id: str
    entity_name: str
    pinyin_text: str
    similarity_score: float
    match_type: str  # 'exact', 'pinyin', 'fuzzy'
    original_query: str
    pinyin_query: str


@dataclass
class EntityRecord:
    """实体记录"""
    entity_id: str
    entity_name: str
    pinyin_text: str
    pinyin_vector: np.ndarray
    metadata: Dict[str, Any] = None


class PinyinNormalizer:
    """拼音标准化器"""
    
    def __init__(self):
        """初始化拼音标准化器"""
        
        # 声母映射表（解决发音相似问题）
        self.initials_mapping = {
            'zh': 'z',  # zh -> z (如：周 -> zou)
            'ch': 'c',  # ch -> c
            'sh': 's',  # sh -> s
            'z': 'zh',  # 反向映射
            'c': 'ch',
            's': 'sh',
        }
        
        # 韵母映射表（解决韵母相似问题）
        self.finals_mapping = {
            'ou': 'uo',  # ou -> uo
            'uo': 'ou',  # uo -> ou
            'ing': 'in',  # ing -> in
            'in': 'ing',  # in -> ing
        }
        
        # 常见多音字处理
        self.polyphone_dict = {
            '周': ['zhou', 'zou'],
            '张': ['zhang', 'zang'],
            '王': ['wang', 'wan'],
            '李': ['li'],
            '陈': ['chen', 'cen'],
            '刘': ['liu', 'lou'],
            '杨': ['yang', 'yan'],
            '黄': ['huang', 'wan'],
            '赵': ['zhao', 'zao'],
            '吴': ['wu'],
            '萍': ['ping', 'pin'],
            '敏': ['min', 'ming'],
            '华': ['hua', 'wa'],
            '丽': ['li'],
            '红': ['hong', 'gong'],
        }
    
    def text_to_pinyin(self, text: str, with_tone: bool = False) -> str:
        """
        将中文文本转换为拼音
        
        Args:
            text: 中文文本
            with_tone: 是否包含声调
        
        Returns:
            拼音字符串
        """
        
        if not text:
            return ""
        
        # 清理文本
        cleaned_text = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
        
        if with_tone:
            # 带声调的拼音
            pinyin_list = pinyin(cleaned_text, style=Style.TONE)
        else:
            # 不带声调的拼音
            pinyin_list = pinyin(cleaned_text, style=Style.NORMAL)
        
        # 提取拼音字符串
        pinyin_text = ' '.join([item[0] for item in pinyin_list])
        
        return pinyin_text.lower()
    
    def generate_pinyin_variants(self, text: str) -> List[str]:
        """
        生成拼音变体（处理多音字和发音相似）
        
        Args:
            text: 中文文本
        
        Returns:
            拼音变体列表
        """
        
        variants = set()
        
        # 1. 标准拼音
        standard_pinyin = self.text_to_pinyin(text)
        variants.add(standard_pinyin)
        
        # 2. 处理多音字
        for char in text:
            if char in self.polyphone_dict:
                for variant_pinyin in self.polyphone_dict[char]:
                    # 替换对应字符的拼音
                    char_pinyin = self.text_to_pinyin(char)
                    variant_text = standard_pinyin.replace(char_pinyin, variant_pinyin)
                    variants.add(variant_text)
        
        # 3. 处理声母相似
        for variant in list(variants):
            for old_initial, new_initial in self.initials_mapping.items():
                if old_initial in variant:
                    new_variant = variant.replace(old_initial, new_initial)
                    variants.add(new_variant)
        
        return list(variants)
    
    def calculate_pinyin_similarity(self, pinyin1: str, pinyin2: str) -> float:
        """
        计算两个拼音字符串的相似度
        
        Args:
            pinyin1: 第一个拼音字符串
            pinyin2: 第二个拼音字符串
        
        Returns:
            相似度分数 (0-1)
        """
        
        if not pinyin1 or not pinyin2:
            return 0.0
        
        # 1. 精确匹配
        if pinyin1 == pinyin2:
            return 1.0
        
        # 2. 使用编辑距离计算相似度
        similarity = difflib.SequenceMatcher(None, pinyin1, pinyin2).ratio()
        
        # 3. 考虑声母韵母相似度
        syllables1 = pinyin1.split()
        syllables2 = pinyin2.split()
        
        if len(syllables1) == len(syllables2):
            syllable_similarities = []
            for s1, s2 in zip(syllables1, syllables2):
                syll_sim = self._calculate_syllable_similarity(s1, s2)
                syllable_similarities.append(syll_sim)
            
            # 平均音节相似度
            avg_syllable_sim = np.mean(syllable_similarities)
            
            # 综合相似度
            similarity = max(similarity, avg_syllable_sim)
        
        return similarity
    
    def _calculate_syllable_similarity(self, syll1: str, syll2: str) -> float:
        """计算单个音节相似度"""
        
        if syll1 == syll2:
            return 1.0
        
        # 检查声母韵母映射
        for old_part, new_part in {**self.initials_mapping, **self.finals_mapping}.items():
            if old_part in syll1 and new_part in syll2:
                return 0.8  # 声母/韵母相似
            if old_part in syll2 and new_part in syll1:
                return 0.8
        
        # 编辑距离
        return difflib.SequenceMatcher(None, syll1, syll2).ratio()


class PinyinVectorSearchEngine:
    """拼音向量搜索引擎"""
    
    def __init__(self, 
                 embedding_model: Optional[BGEEmbeddingModel] = None,
                 similarity_threshold: float = 0.7):
        """
        初始化拼音向量搜索引擎
        
        Args:
            embedding_model: BGE嵌入模型
            similarity_threshold: 相似度阈值
        """
        
        self.embedding_model = embedding_model or get_bge_model()
        self.similarity_threshold = similarity_threshold
        self.pinyin_normalizer = PinyinNormalizer()
        
        # 实体存储
        self.entities: Dict[str, EntityRecord] = {}
        self.entity_vectors: List[np.ndarray] = []
        self.entity_ids: List[str] = []
        
        print(f"🔧 初始化拼音向量搜索引擎")
        print(f"   相似度阈值: {similarity_threshold}")
    
    def add_entity(self, entity_id: str, entity_name: str, metadata: Dict[str, Any] = None):
        """
        添加可搜索实体
        
        Args:
            entity_id: 实体ID
            entity_name: 实体名称（中文）
            metadata: 元数据
        """
        
        try:
            # 转换为拼音
            pinyin_text = self.pinyin_normalizer.text_to_pinyin(entity_name)
            
            # 生成拼音变体
            pinyin_variants = self.pinyin_normalizer.generate_pinyin_variants(entity_name)
            
            # 向量化所有拼音变体
            all_pinyin_texts = [pinyin_text] + pinyin_variants
            pinyin_vectors = []
            
            for py_text in all_pinyin_texts:
                if py_text:  # 确保不为空
                    vector = self.embedding_model.encode_text(py_text)
                    pinyin_vectors.append(vector)
            
            # 计算平均向量
            if pinyin_vectors:
                avg_vector = np.mean(pinyin_vectors, axis=0)
            else:
                # 如果拼音为空，使用原文向量
                avg_vector = self.embedding_model.encode_text(entity_name)
            
            # 创建实体记录
            entity_record = EntityRecord(
                entity_id=entity_id,
                entity_name=entity_name,
                pinyin_text=pinyin_text,
                pinyin_vector=avg_vector,
                metadata=metadata or {}
            )
            
            # 存储实体
            self.entities[entity_id] = entity_record
            self.entity_vectors.append(avg_vector)
            self.entity_ids.append(entity_id)
            
            print(f"✅ 添加实体: {entity_name} -> {pinyin_text}")
            
        except Exception as e:
            print(f"❌ 添加实体失败 {entity_name}: {e}")
    
    def batch_add_entities(self, entities: List[Dict[str, Any]]):
        """
        批量添加实体
        
        Args:
            entities: 实体列表，每个元素包含 entity_id, entity_name, metadata
        """
        
        print(f"🔄 正在批量添加 {len(entities)} 个实体...")
        
        for entity_data in entities:
            self.add_entity(
                entity_id=entity_data.get('entity_id'),
                entity_name=entity_data.get('entity_name'),
                metadata=entity_data.get('metadata', {})
            )
        
        print(f"✅ 批量添加完成，总计 {len(self.entities)} 个实体")
    
    def search(self, query: str, top_k: int = 10, enable_fuzzy: bool = True) -> List[PinyinSearchResult]:
        """
        拼音向量搜索
        
        Args:
            query: 查询文本（可能是语音识别结果）
            top_k: 返回结果数量
            enable_fuzzy: 是否启用模糊匹配
        
        Returns:
            搜索结果列表
        """
        
        if not self.entities:
            return []
        
        try:
            print(f"🔍 拼音向量搜索: {query}")
            
            results = []
            
            # 1. 精确文字匹配
            exact_results = self._exact_text_search(query)
            results.extend(exact_results)
            
            # 2. 拼音向量搜索
            pinyin_results = self._pinyin_vector_search(query, top_k)
            results.extend(pinyin_results)
            
            # 3. 模糊拼音匹配（如果启用）
            if enable_fuzzy:
                fuzzy_results = self._fuzzy_pinyin_search(query)
                results.extend(fuzzy_results)
            
            # 4. 去重和排序
            unique_results = self._deduplicate_and_rank(results, top_k)
            
            print(f"🎯 找到 {len(unique_results)} 个匹配结果")
            
            return unique_results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def _exact_text_search(self, query: str) -> List[PinyinSearchResult]:
        """精确文字匹配"""
        
        results = []
        
        for entity_id, entity in self.entities.items():
            if query in entity.entity_name or entity.entity_name in query:
                result = PinyinSearchResult(
                    entity_id=entity_id,
                    entity_name=entity.entity_name,
                    pinyin_text=entity.pinyin_text,
                    similarity_score=1.0,
                    match_type='exact',
                    original_query=query,
                    pinyin_query=self.pinyin_normalizer.text_to_pinyin(query)
                )
                results.append(result)
        
        return results
    
    def _pinyin_vector_search(self, query: str, top_k: int) -> List[PinyinSearchResult]:
        """拼音向量搜索"""
        
        if not self.entity_vectors:
            return []
        
        try:
            # 将查询转换为拼音
            query_pinyin = self.pinyin_normalizer.text_to_pinyin(query)
            
            # 生成查询拼音变体
            query_variants = self.pinyin_normalizer.generate_pinyin_variants(query)
            all_queries = [query_pinyin] + query_variants
            
            # 向量化所有查询变体
            query_vectors = []
            for q_text in all_queries:
                if q_text:
                    vector = self.embedding_model.encode_text(q_text)
                    query_vectors.append(vector)
            
            if not query_vectors:
                return []
            
            # 计算平均查询向量
            avg_query_vector = np.mean(query_vectors, axis=0)
            
            # 计算与所有实体的相似度
            entity_matrix = np.stack(self.entity_vectors)
            similarities = self.embedding_model.compute_similarity(
                avg_query_vector.reshape(1, -1), 
                entity_matrix
            )[0]
            
            # 获取top-k结果
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # 多取一些用于后续筛选
            
            results = []
            for idx in top_indices:
                if similarities[idx] >= self.similarity_threshold:
                    entity_id = self.entity_ids[idx]
                    entity = self.entities[entity_id]
                    
                    result = PinyinSearchResult(
                        entity_id=entity_id,
                        entity_name=entity.entity_name,
                        pinyin_text=entity.pinyin_text,
                        similarity_score=float(similarities[idx]),
                        match_type='pinyin',
                        original_query=query,
                        pinyin_query=query_pinyin
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ 拼音向量搜索失败: {e}")
            return []
    
    def _fuzzy_pinyin_search(self, query: str) -> List[PinyinSearchResult]:
        """模糊拼音搜索"""
        
        query_pinyin = self.pinyin_normalizer.text_to_pinyin(query)
        
        results = []
        
        for entity_id, entity in self.entities.items():
            # 计算拼音相似度
            pinyin_similarity = self.pinyin_normalizer.calculate_pinyin_similarity(
                query_pinyin, entity.pinyin_text
            )
            
            if pinyin_similarity >= self.similarity_threshold:
                result = PinyinSearchResult(
                    entity_id=entity_id,
                    entity_name=entity.entity_name,
                    pinyin_text=entity.pinyin_text,
                    similarity_score=pinyin_similarity,
                    match_type='fuzzy',
                    original_query=query,
                    pinyin_query=query_pinyin
                )
                results.append(result)
        
        return results
    
    def _deduplicate_and_rank(self, results: List[PinyinSearchResult], top_k: int) -> List[PinyinSearchResult]:
        """去重和排序"""
        
        # 按entity_id去重，保留最高分
        unique_results = {}
        
        for result in results:
            entity_id = result.entity_id
            if entity_id not in unique_results or result.similarity_score > unique_results[entity_id].similarity_score:
                unique_results[entity_id] = result
        
        # 排序并返回top-k
        sorted_results = sorted(unique_results.values(), key=lambda x: x.similarity_score, reverse=True)
        
        return sorted_results[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取搜索引擎统计信息"""
        
        return {
            "total_entities": len(self.entities),
            "vector_dimension": self.embedding_model.embedding_dim,
            "similarity_threshold": self.similarity_threshold,
            "entity_list": [entity.entity_name for entity in self.entities.values()]
        }


# 测试代码
if __name__ == "__main__":
    # 创建拼音向量搜索引擎
    search_engine = PinyinVectorSearchEngine()
    
    # 添加测试实体（常见人名）
    test_entities = [
        {"entity_id": "person_001", "entity_name": "周萍", "metadata": {"type": "person"}},
        {"entity_id": "person_002", "entity_name": "张敏", "metadata": {"type": "person"}},
        {"entity_id": "person_003", "entity_name": "李华", "metadata": {"type": "person"}},
        {"entity_id": "person_004", "entity_name": "王红", "metadata": {"type": "person"}},
        {"entity_id": "person_005", "entity_name": "陈丽", "metadata": {"type": "person"}},
        {"entity_id": "person_006", "entity_name": "刘敏", "metadata": {"type": "person"}},
        {"entity_id": "person_007", "entity_name": "杨华", "metadata": {"type": "person"}},
        {"entity_id": "person_008", "entity_name": "黄萍", "metadata": {"type": "person"}},
    ]
    
    # 批量添加实体
    search_engine.batch_add_entities(test_entities)
    
    # 测试搜索
    test_queries = [
        "邹萍",  # 错误发音 -> 应该找到"周萍"
        "张民",  # 错误发音 -> 应该找到"张敏"
        "李华",  # 正确发音
        "忘红",  # 错误发音 -> 应该找到"王红"
    ]
    
    print("\n" + "="*60)
    print("拼音向量搜索测试")
    print("="*60)
    
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        results = search_engine.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.entity_name} (ID: {result.entity_id})")
            print(f"     拼音: {result.pinyin_text}")
            print(f"     相似度: {result.similarity_score:.3f}")
            print(f"     匹配类型: {result.match_type}")
    
    # 显示统计信息
    print(f"\n📊 搜索引擎统计:")
    stats = search_engine.get_statistics()
    for key, value in stats.items():
        if key != "entity_list":
            print(f"   {key}: {value}") 