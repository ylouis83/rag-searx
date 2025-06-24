#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGE-large-zh-v1.5 向量化模型
基于sentence-transformers的中文语义向量化
"""

import os
import time
import numpy as np
from typing import List, Union, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path


class BGEEmbeddingModel:
    """BGE-large-zh-v1.5 中文向量化模型"""
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_seq_length: int = 512
    ):
        """
        初始化BGE向量化模型
        
        Args:
            model_name: 模型名称，默认BAAI/bge-large-zh-v1.5
            device: 计算设备，None为自动选择
            cache_dir: 模型缓存目录
            max_seq_length: 最大序列长度
        """
        
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        
        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon GPU
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # BGE-large-zh-v1.5输出1024维向量
        self.embedding_dim = 1024
        
        # 延迟加载模型
        self.model = None
        self._model_loaded = False
        
        print(f"🔧 初始化BGE向量化模型")
        print(f"   模型名称: {self.model_name}")
        print(f"   计算设备: {self.device}")
        print(f"   向量维度: {self.embedding_dim}")
        print(f"   序列长度: {self.max_seq_length}")
    
    def _load_model(self):
        """延迟加载模型"""
        
        if self._model_loaded:
            return
        
        try:
            print(f"📥 正在加载BGE模型...")
            start_time = time.time()
            
            # 使用sentence-transformers加载
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            
            # 设置最大序列长度
            self.model.max_seq_length = self.max_seq_length
            
            # 获取实际向量维度
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            load_time = time.time() - start_time
            print(f"✅ BGE模型加载完成: {load_time:.2f}秒")
            print(f"   实际向量维度: {self.embedding_dim}")
            
            self._model_loaded = True
            
        except Exception as e:
            print(f"❌ BGE模型加载失败: {e}")
            raise
    
    def encode_text(self, text: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        对文本进行向量化编码
        
        Args:
            text: 单个文本或文本列表
            **kwargs: 其他编码参数
        
        Returns:
            向量化结果，shape为(batch_size, embedding_dim)
        """
        
        # 确保模型已加载
        self._load_model()
        
        # 处理输入格式
        if isinstance(text, str):
            texts = [text]
            single_text = True
        else:
            texts = text
            single_text = False
        
        # 预处理文本
        processed_texts = self._preprocess_texts(texts)
        
        try:
            # 执行向量化
            start_time = time.time()
            
            embeddings = self.model.encode(
                processed_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2归一化
                show_progress_bar=len(processed_texts) > 10,
                batch_size=32,
                **kwargs
            )
            
            encode_time = time.time() - start_time
            
            if len(processed_texts) > 1:
                print(f"⚡ 批量编码 {len(processed_texts)} 条文本: {encode_time:.3f}秒")
                print(f"   处理速度: {len(processed_texts)/encode_time:.1f} 条/秒")
            
            # 返回结果
            if single_text:
                return embeddings[0]
            else:
                return embeddings
                
        except Exception as e:
            print(f"❌ 文本编码失败: {e}")
            raise
    
    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """
        对查询文本进行向量化编码
        
        Args:
            query: 查询文本
            **kwargs: 其他编码参数
        
        Returns:
            查询向量
        """
        
        # BGE模型对查询文本的特殊处理
        instruction = ""
        formatted_query = f"{instruction}{query}"
        
        return self.encode_text(formatted_query, **kwargs)
    
    def encode_document(self, document: str, **kwargs) -> np.ndarray:
        """
        对文档进行向量化编码
        
        Args:
            document: 文档文本
            **kwargs: 其他编码参数
        
        Returns:
            文档向量
        """
        
        # 直接编码文档
        return self.encode_text(document, **kwargs)
    
    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        计算向量相似度
        
        Args:
            embeddings1: 第一组向量
            embeddings2: 第二组向量
        
        Returns:
            相似度矩阵
        """
        
        # 确保是2D数组
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        # 计算余弦相似度
        similarity = np.dot(embeddings1, embeddings2.T)
        
        return similarity
    
    def batch_encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        批量文本向量化
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
        
        Returns:
            向量化结果矩阵
        """
        
        # 确保模型已加载
        self._load_model()
        
        print(f"🔄 正在批量编码 {len(texts)} 条文本...")
        
        # 预处理文本
        processed_texts = self._preprocess_texts(texts)
        
        try:
            start_time = time.time()
            
            embeddings = self.model.encode(
                processed_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                batch_size=batch_size
            )
            
            encode_time = time.time() - start_time
            
            print(f"✅ 批量编码完成: {encode_time:.2f}秒")
            print(f"   处理速度: {len(texts)/encode_time:.1f} 条/秒")
            print(f"   输出形状: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            print(f"❌ 批量编码失败: {e}")
            raise
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        预处理文本列表
        
        Args:
            texts: 原始文本列表
        
        Returns:
            预处理后的文本列表
        """
        
        processed = []
        
        for text in texts:
            # 清理文本
            cleaned_text = self._clean_text(text)
            
            # 限制长度
            if len(cleaned_text) > self.max_seq_length * 2:  # 预留一些空间
                cleaned_text = cleaned_text[:self.max_seq_length * 2]
            
            processed.append(cleaned_text)
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """
        清理单个文本
        
        Args:
            text: 原始文本
        
        Returns:
            清理后的文本
        """
        
        if not text or not text.strip():
            return ""
        
        # 规范化空白字符
        text = " ".join(text.split())
        
        # 可选：过滤特殊字符（暂时注释掉）
        # text = re.sub(r'[^\u4e00-\u9fff\w\s\.\!\?\,\;\:\-\(\)]', ' ', text)
        
        return text.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        
        info = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_seq_length": self.max_seq_length,
            "device": self.device,
            "model_loaded": self._model_loaded
        }
        
        if self._model_loaded:
            info["model_config"] = str(self.model)
        
        return info
    
    def save_embeddings(self, embeddings: np.ndarray, file_path: str):
        """
        保存向量到文件
        
        Args:
            embeddings: 向量矩阵
            file_path: 保存路径
        """
        
        try:
            np.save(file_path, embeddings)
            print(f"💾 向量已保存: {file_path}")
        except Exception as e:
            print(f"❌ 向量保存失败: {e}")
            raise
    
    def load_embeddings(self, file_path: str) -> np.ndarray:
        """
        从文件加载向量
        
        Args:
            file_path: 文件路径
        
        Returns:
            向量矩阵
        """
        
        try:
            embeddings = np.load(file_path)
            print(f"📂 向量已加载: {file_path}")
            print(f"   向量形状: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"❌ 向量加载失败: {e}")
            raise


# 全局模型实例
_bge_model: Optional[BGEEmbeddingModel] = None


def get_bge_model(
    model_name: str = "BAAI/bge-large-zh-v1.5",
    device: Optional[str] = None,
    **kwargs
) -> BGEEmbeddingModel:
    """
    获取BGE模型单例
    
    Args:
        model_name: 模型名称
        device: 计算设备
        **kwargs: 其他参数
    
    Returns:
        BGE模型实例
    """
    
    global _bge_model
    
    if _bge_model is None:
        _bge_model = BGEEmbeddingModel(
            model_name=model_name,
            device=device,
            **kwargs
        )
    
    return _bge_model


# 测试代码
if __name__ == "__main__":
    # 创建BGE模型
    bge = BGEEmbeddingModel()
    
    # 测试文本
    test_texts = [
        "糖尿病是一种慢性疾病",
        "血糖控制对糖尿病患者很重要",
        "合理饮食有助于血糖稳定"
    ]
    
    # 单文本编码测试
    print("\n📝 单文本编码测试")
    single_embedding = bge.encode_text(test_texts[0])
    print(f"向量形状: {single_embedding.shape}")
    print(f"向量模长: {np.linalg.norm(single_embedding):.4f}")
    
    # 批量编码测试
    print("\n📚 批量编码测试")
    batch_embeddings = bge.batch_encode(test_texts)
    print(f"批量向量形状: {batch_embeddings.shape}")
    
    # 相似度计算测试
    print("\n🔗 相似度计算测试")
    similarities = bge.compute_similarity(batch_embeddings[0:1], batch_embeddings)
    print(f"相似度结果: {similarities[0]}")
    
    # 查询编码测试
    print("\n❓ 查询编码测试")
    query = "如何预防糖尿病"
    query_embedding = bge.encode_query(query)
    print(f"查询向量形状: {query_embedding.shape}")
    
    # 模型信息测试
    print("\n📊 模型信息")
    info = bge.get_model_info()
    for key, value in info.items():
        if key != "model_config":
            print(f"   {key}: {value}") 