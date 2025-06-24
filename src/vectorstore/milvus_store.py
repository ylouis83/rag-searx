"""
Milvus

Milvus
"""

import json
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections, 
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType,
    utility
)

from config.settings import get_settings
from ..utils.logger import get_logger
from .base import VectorStore, SearchResult
from ..api.schemas import VectorDBMetadata, ChunkData

# 
settings = get_settings()
logger = get_logger(__name__)


class MilvusStore(VectorStore):
    """Milvus"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.connection_alias = "default"
        self.collections: Dict[str, Collection] = {}
        
    async def initialize(self) -> None:
        """Milvus"""
        try:
            # Milvus
            connections.connect(
                alias=self.connection_alias,
                host=self.settings.vector_db.host,
                port=self.settings.vector_db.port,
                user=self.settings.vector_db.username or "",
                password=self.settings.vector_db.password or ""
            )
            
            self.logger.info(f"Milvus: {self.settings.vector_db.host}:{self.settings.vector_db.port}")
            
            # 
            collection_name = self.settings.vector_db.milvus_collection_name
            if not utility.has_collection(collection_name):
                await self.create_collection(collection_name, self.settings.embedding.dimension)
            else:
                # 
                self.collections[collection_name] = Collection(collection_name)
                self.collections[collection_name].load()
                self.logger.info(f": {collection_name}")
                
        except Exception as e:
            self.logger.error(f"Milvus: {e}")
            raise
    
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """Milvus"""
        try:
            # Schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=128,
                    is_primary=True,
                    auto_id=False
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=65535
                ),
                FieldSchema(
                    name="media_type",
                    dtype=DataType.VARCHAR,
                    max_length=32
                ),
                FieldSchema(
                    name="book_title",
                    dtype=DataType.VARCHAR,
                    max_length=256
                ),
                FieldSchema(
                    name="source_text",
                    dtype=DataType.VARCHAR,
                    max_length=1024
                ),
                FieldSchema(
                    name="source_url",
                    dtype=DataType.VARCHAR,
                    max_length=512
                ),
                FieldSchema(
                    name="chapter_number",
                    dtype=DataType.INT32
                ),
                FieldSchema(
                    name="chapter_title",
                    dtype=DataType.VARCHAR,
                    max_length=256
                ),
                FieldSchema(
                    name="page_number",
                    dtype=DataType.INT32
                ),
                FieldSchema(
                    name="metadata_json",
                    dtype=DataType.VARCHAR,
                    max_length=65535
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dimension
                )
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description=f"RAG-SearX: {collection_name}"
            )
            
            # 
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.connection_alias
            )
            
            # 
            index_params = {
                "index_type": self.settings.vector_db.milvus_index_type,
                "metric_type": self.settings.vector_db.milvus_metric_type,
                "params": {"nlist": 1024}
            }
            
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            # 
            collection.load()
            
            self.collections[collection_name] = collection
            self.logger.info(f"Milvus: {collection_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Milvus: {e}")
            return False
    
    async def insert_vectors(
        self, 
        collection_name: str,
        chunks: List[ChunkData]
    ) -> bool:
        """Milvus"""
        try:
            if collection_name not in self.collections:
                self.logger.error(f": {collection_name}")
                return False
                
            collection = self.collections[collection_name]
            
            # 
            data = []
            for chunk in chunks:
                if not chunk.embeddings:
                    self.logger.warning(f" {chunk.chunk_id} ")
                    continue
                    
                data.append({
                    "id": chunk.chunk_id,
                    "content": chunk.content,
                    "media_type": chunk.metadata.media_type,
                    "book_title": chunk.metadata.book_title,
                    "source_text": chunk.metadata.source_text,
                    "source_url": chunk.metadata.source_url or "",
                    "chapter_number": chunk.metadata.context_link.chapter_number,
                    "chapter_title": chunk.metadata.context_link.chapter_title,
                    "page_number": chunk.metadata.context_link.page_number,
                    "metadata_json": json.dumps(chunk.metadata.dict(), ensure_ascii=False),
                    "embedding": chunk.embeddings
                })
            
            if not data:
                self.logger.warning("")
                return False
            
            # 
            mr = collection.insert(data)
            collection.flush()
            
            self.logger.info(f" {len(data)}  {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f": {e}")
            return False
    
    async def search_vectors(
        self, 
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None
    ) -> List[SearchResult]:
        """"""
        try:
            if collection_name not in self.collections:
                self.logger.error(f": {collection_name}")
                return []
                
            collection = self.collections[collection_name]
            
            # 
            search_params = {
                "metric_type": self.settings.vector_db.milvus_metric_type,
                "params": {"nprobe": 10}
            }
            
            # 
            output_fields = [
                "content", "media_type", "book_title", "source_text", 
                "source_url", "chapter_number", "chapter_title", 
                "page_number", "metadata_json"
            ]
            
            # 
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields
            )
            
            # 
            search_results = []
            for hits in results:
                for hit in hits:
                    try:
                        # 
                        metadata_dict = json.loads(hit.entity.get("metadata_json", "{}"))
                        metadata = VectorDBMetadata(**metadata_dict)
                        
                        search_result = SearchResult(
                            id=hit.id,
                            content=hit.entity.get("content", ""),
                            metadata=metadata,
                            score=hit.score
                        )
                        search_results.append(search_result)
                        
                    except Exception as e:
                        self.logger.warning(f": {e}")
                        continue
            
            self.logger.info(f" {len(search_results)} ")
            return search_results
            
        except Exception as e:
            self.logger.error(f": {e}")
            return []
    
    async def delete_vectors(
        self, 
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """"""
        try:
            if collection_name not in self.collections:
                self.logger.error(f": {collection_name}")
                return False
                
            collection = self.collections[collection_name]
            
            # 
            ids_str = "','".join(ids)
            expr = f"id in ['{ids_str}']"
            
            # 
            collection.delete(expr)
            collection.flush()
            
            self.logger.info(f" {len(ids)} ")
            return True
            
        except Exception as e:
            self.logger.error(f": {e}")
            return False
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """"""
        try:
            if collection_name not in self.collections:
                return {}
                
            collection = self.collections[collection_name]
            
            # 
            stats = collection.get_stats()
            schema = collection.schema
            
            info = {
                "name": collection_name,
                "description": schema.description,
                "num_entities": stats.num_entities if hasattr(stats, 'num_entities') else 0,
                "fields": [
                    {
                        "name": field.name,
                        "type": field.dtype.name,
                        "is_primary": field.is_primary
                    }
                    for field in schema.fields
                ],
                "indexes": [
                    {
                        "field_name": index.field_name,
                        "index_type": index.params.get("index_type"),
                        "metric_type": index.params.get("metric_type")
                    }
                    for index in collection.indexes
                ]
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f": {e}")
            return {}
    
    async def close(self) -> None:
        """Milvus"""
        try:
            # 
            for collection in self.collections.values():
                collection.release()
            
            # 
            connections.disconnect(self.connection_alias)
            
            self.logger.info("Milvus")
            
        except Exception as e:
            self.logger.error(f"Milvus: {e}")
    
    async def hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        text_filter: Optional[str] = None,
        media_type_filter: Optional[str] = None,
        book_title_filter: Optional[str] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """"""
        
        try:
            # 
            filter_conditions = []
            
            if text_filter:
                filter_conditions.append(f"content like '%{text_filter}%'")
            
            if media_type_filter:
                filter_conditions.append(f"media_type == '{media_type_filter}'")
                
            if book_title_filter:
                filter_conditions.append(f"book_title == '{book_title_filter}'")
            
            filter_expr = " and ".join(filter_conditions) if filter_conditions else None
            
            return await self.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
                filter_expr=filter_expr
            )
            
        except Exception as e:
            self.logger.error(f": {e}")
            return []


# Milvus
_milvus_store: Optional[MilvusStore] = None


async def get_milvus_store() -> MilvusStore:
    """Milvus"""
    global _milvus_store
    
    if _milvus_store is None:
        _milvus_store = MilvusStore()
        await _milvus_store.initialize()
    
    return _milvus_store


async def close_milvus_store():
    """Milvus"""
    global _milvus_store
    
    if _milvus_store is not None:
        await _milvus_store.close()
        _milvus_store = None 