"""



"""

import asyncio
from typing import List, Union, Optional, Dict, Any
from abc import ABC, abstractmethod
import httpx
import numpy as np

from config.settings import get_settings
from ..utils.logger import get_logger

# 
settings = get_settings()
logger = get_logger(__name__)


class EmbeddingModel(ABC):
    """"""
    
    @abstractmethod
    async def encode_text(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """"""
        pass
    
    @abstractmethod
    async def encode_image(self, image_path: str) -> List[float]:
        """"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """"""
        pass


class DashScopeEmbedding(EmbeddingModel):
    """"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.api_key = self.settings.embedding.api_key
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        self.model = self.settings.embedding.model or "text-embedding-v1"
        self.dimension = self.settings.embedding.dimension
        
        if not self.api_key:
            raise ValueError("API")
    
    async def encode_text(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """"""
        
        try:
            # 
            if isinstance(texts, str):
                text_list = [texts]
                return_single = True
            else:
                text_list = texts
                return_single = False
            
            self.logger.info(f": {len(text_list)}")
            
            # 
            batch_size = self.settings.embedding.batch_size
            all_embeddings = []
            
            for i in range(0, len(text_list), batch_size):
                batch_texts = text_list[i:i + batch_size]
                batch_embeddings = await self._encode_text_batch(batch_texts)
                all_embeddings.extend(batch_embeddings)
            
            if return_single:
                return all_embeddings[0]
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f": {e}")
            raise
    
    async def _encode_text_batch(self, texts: List[str]) -> List[List[float]]:
        """"""
        
        try:
            # 
            payload = {
                "model": self.model,
                "input": {
                    "texts": texts
                },
                "parameters": {
                    "text_type": "document"
                }
            }
            
            # 
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # 
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code != 200:
                    error_msg = f"DashScopeAPI: {response.status_code} - {response.text}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                
                # 
                result = response.json()
                
                if result.get("status_code") != 200:
                    error_msg = f"DashScopeAPI: {result.get('message', '')}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                
                # 
                output = result.get("output", {})
                embeddings = output.get("embeddings", [])
                
                if not embeddings or len(embeddings) != len(texts):
                    raise Exception("")
                
                # 
                vectors = []
                for embedding in embeddings:
                    vector = embedding.get("embedding", [])
                    if not vector:
                        raise Exception("")
                    vectors.append(vector)
                
                return vectors
                
        except Exception as e:
            self.logger.error(f": {e}")
            raise
    
    async def encode_image(self, image_path: str) -> List[float]:
        """DashScopeCLIP"""
        
        try:
            # CLIP
            from sentence_transformers import SentenceTransformer
            import PIL.Image
            
            # CLIP
            model = SentenceTransformer('clip-ViT-B-32')
            
            # 
            image = PIL.Image.open(image_path)
            
            # 
            image_embedding = model.encode([image])
            
            return image_embedding[0].tolist()
            
        except Exception as e:
            self.logger.error(f": {e}")
            # fallback
            return [0.0] * self.get_dimension()
    
    def get_dimension(self) -> int:
        """"""
        return self.dimension


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.api_key = self.settings.llm.api_key  # LLMAPI
        self.base_url = self.settings.llm.base_url or "https://api.openai.com/v1"
        self.model = self.settings.embedding.openai_embedding_model
        self.dimension = self.settings.embedding.openai_embedding_dimension
        
        if not self.api_key:
            raise ValueError("OpenAI API")
    
    async def encode_text(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """"""
        
        try:
            # 
            if isinstance(texts, str):
                text_list = [texts]
                return_single = True
            else:
                text_list = texts
                return_single = False
            
            self.logger.info(f": {len(text_list)}")
            
            # 
            payload = {
                "model": self.model,
                "input": text_list,
                "encoding_format": "float"
            }
            
            # 
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code != 200:
                    error_msg = f"OpenAIAPI: {response.status_code} - {response.text}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                
                # 
                result = response.json()
                
                # 
                embeddings = []
                for item in result.get("data", []):
                    embeddings.append(item.get("embedding", []))
                
                if return_single:
                    return embeddings[0]
                return embeddings
                
        except Exception as e:
            self.logger.error(f"OpenAI: {e}")
            raise
    
    async def encode_image(self, image_path: str) -> List[float]:
        """CLIP"""
        
        try:
            # sentence-transformersCLIP
            from sentence_transformers import SentenceTransformer
            import PIL.Image
            
            # CLIP
            model = SentenceTransformer('clip-ViT-B-32')
            
            # 
            image = PIL.Image.open(image_path)
            
            # 
            image_embedding = model.encode([image])
            
            return image_embedding[0].tolist()
            
        except Exception as e:
            self.logger.error(f": {e}")
            # fallback
            return [0.0] * self.get_dimension()
    
    def get_dimension(self) -> int:
        """"""
        return self.dimension


class HuggingFaceEmbedding(EmbeddingModel):
    """HuggingFace"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.model_name = self.settings.embedding.hf_embedding_model
        self.dimension = self.settings.embedding.dimension
        self.model = None
        
    async def _load_model(self):
        """"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                self.logger.info(f"HuggingFace: {self.model_name}")
            except Exception as e:
                self.logger.error(f"HuggingFace: {e}")
                raise
    
    async def encode_text(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """"""
        
        try:
            await self._load_model()
            
            # 
            if isinstance(texts, str):
                text_list = [texts]
                return_single = True
            else:
                text_list = texts
                return_single = False
            
            self.logger.info(f": {len(text_list)}")
            
            # 
            embeddings = self.model.encode(text_list, convert_to_tensor=False)
            
            # 
            if return_single:
                return embeddings[0].tolist()
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            self.logger.error(f"HuggingFace: {e}")
            raise
    
    async def encode_image(self, image_path: str) -> List[float]:
        """CLIP"""
        
        try:
            from sentence_transformers import SentenceTransformer
            import PIL.Image
            
            # CLIP
            clip_model = SentenceTransformer('clip-ViT-B-32')
            
            # 
            image = PIL.Image.open(image_path)
            
            # 
            image_embedding = clip_model.encode([image])
            
            return image_embedding[0].tolist()
            
        except Exception as e:
            self.logger.error(f": {e}")
            # fallback
            return [0.0] * self.get_dimension()
    
    def get_dimension(self) -> int:
        """"""
        return self.dimension


class JinaEmbedding(EmbeddingModel):
    """Jina AI"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.api_key = self.settings.embedding.api_key
        self.base_url = "https://api.jina.ai/v1/embeddings"
        self.model = self.settings.embedding.model
        self.dimension = self.settings.embedding.dimension
        
        if not self.api_key:
            raise ValueError("Jina AI API")
    
    async def encode_text(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """"""
        
        try:
            # 
            if isinstance(texts, str):
                text_list = [texts]
                return_single = True
            else:
                text_list = texts
                return_single = False
            
            self.logger.info(f": {len(text_list)}")
            
            # 
            payload = {
                "model": self.model,
                "input": text_list,
                "encoding_format": "float"
            }
            
            # 
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code != 200:
                    error_msg = f"Jina AIAPI: {response.status_code} - {response.text}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                
                # 
                result = response.json()
                
                # 
                embeddings = []
                for item in result.get("data", []):
                    embeddings.append(item.get("embedding", []))
                
                if return_single:
                    return embeddings[0]
                return embeddings
                
        except Exception as e:
            self.logger.error(f"Jina AI: {e}")
            raise
    
    async def encode_image(self, image_path: str) -> List[float]:
        """CLIP"""
        
        try:
            from sentence_transformers import SentenceTransformer
            import PIL.Image
            
            # CLIP
            model = SentenceTransformer('clip-ViT-B-32')
            
            # 
            image = PIL.Image.open(image_path)
            
            # 
            image_embedding = model.encode([image])
            
            return image_embedding[0].tolist()
            
        except Exception as e:
            self.logger.error(f": {e}")
            # fallback
            return [0.0] * self.get_dimension()
    
    def get_dimension(self) -> int:
        """"""
        return self.dimension


# 
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """"""
    global _embedding_model
    
    if _embedding_model is None:
        provider = settings.embedding.provider.lower()
        
        if provider == "dashscope":
            _embedding_model = DashScopeEmbedding()
        elif provider == "openai":
            _embedding_model = OpenAIEmbedding()
        elif provider == "huggingface":
            _embedding_model = HuggingFaceEmbedding()
        elif provider == "jina":
            _embedding_model = JinaEmbedding()
        else:
            # HuggingFace
            logger.warning(f": {provider}HuggingFace")
            _embedding_model = HuggingFaceEmbedding()
    
    return _embedding_model


async def encode_texts(texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """"""
    model = get_embedding_model()
    return await model.encode_text(texts)


async def encode_image(image_path: str) -> List[float]:
    """"""
    model = get_embedding_model()
    return await model.encode_image(image_path) 