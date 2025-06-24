"""



"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..api.schemas import VectorDBMetadata, ChunkData


@dataclass
class SearchResult:
    """"""
    id: str
    content: str
    metadata: VectorDBMetadata
    score: float
    embedding: Optional[List[float]] = None


class VectorStore(ABC):
    """"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """"""
        pass
    
    @abstractmethod
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """"""
        pass
    
    @abstractmethod
    async def insert_vectors(
        self, 
        collection_name: str,
        chunks: List[ChunkData]
    ) -> bool:
        """"""
        pass
    
    @abstractmethod
    async def search_vectors(
        self, 
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None
    ) -> List[SearchResult]:
        """"""
        pass
    
    @abstractmethod
    async def delete_vectors(
        self, 
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """"""
        pass
    
    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """"""
        pass 