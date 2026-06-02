#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow

"""

import os
import sys
import time
import uuid
import dashscope
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragflow_integration.video_metadata_schema import VideoMetadata, SemanticChunk, VideoLanguage
from ragflow_integration.bge_embedding import get_bge_model

# API
os.environ['DASHSCOPE_API_KEY'] = os.environ.get("DASHSCOPE_API_KEY", "")
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")


class VideoRAGPipeline:
    """RAG"""
    
    def __init__(
        self,
        collection_name: str = "video_rag_collection",
        embedding_model_name: str = "BAAI/bge-large-zh-v1.5",
        milvus_host: str = "localhost",
        milvus_port: str = "19530"
    ):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        
        self.collection = None
        self.embedding_model = None
        self.milvus_connected = False
        
        # 
        self._init_components()
    
    def _init_components(self):
        """"""
        print(" RAG...")
        
        # BGE
        print(" BGE...")
        self.embedding_model = get_bge_model(model_name=self.embedding_model_name)
        print(" BGE")
        
        # Milvus
        self._connect_milvus()
        
        # 
        self._ensure_collection()
    
    def _connect_milvus(self):
        """Milvus"""
        try:
            print(" Milvus...")
            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port
            )
            self.milvus_connected = True
            print(" Milvus")
        except Exception as e:
            print(f" Milvus: {e}")
            self.milvus_connected = False
    
    def _ensure_collection(self):
        """"""
        if not self.milvus_connected:
            return False
        
        try:
            # 
            if utility.has_collection(self.collection_name):
                print(f"  {self.collection_name} ")
                self.collection = Collection(self.collection_name)
                
                # 
                try:
                    load_state = utility.load_state(self.collection_name)
                    if hasattr(load_state, 'state'):
                        is_loaded = load_state.state.name == "Loaded"
                    else:
                        is_loaded = str(load_state) == "Loaded"
                    
                    if not is_loaded:
                        print(f" ...")
                        self.collection.load()
                        print(f" ")
                    else:
                        print(f" ")
                except Exception as e:
                    print(f"  : {e}")
                    self.collection.load()
                    print(f" ")
                return True
            
            # 
            print(f" : {self.collection_name}")
            
            # Schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
                FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="video_url", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="channel_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=10),
                FieldSchema(name="ai_summary", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="duration_seconds", dtype=DataType.INT64),
                FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),  # summary, chunk, keywords
                FieldSchema(name="chunk_start_time", dtype=DataType.INT64),
                FieldSchema(name="chunk_end_time", dtype=DataType.INT64),
                FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="RAG - "
            )
            
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            # 
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            # 
            self.collection.load()
            print(f"  {self.collection_name} ")
            return True
            
        except Exception as e:
            print(f" : {e}")
            return False
    
    def store_video_metadata(self, video_metadata: VideoMetadata) -> bool:
        """Milvus"""
        if not self.milvus_connected:
            print(" Milvus")
            return False
        
        try:
            print(f" : {video_metadata.title}")
            
            # 
            ids = []
            video_ids = []
            video_urls = []
            titles = []
            channel_names = []
            languages = []
            ai_summaries = []
            keywords_list = []
            categories = []
            duration_seconds_list = []
            content_types = []
            chunk_start_times = []
            chunk_end_times = []
            chunk_texts = []
            embeddings = []
            
            # 1. 
            summary_id = f"{video_metadata.video_id}_summary"
            summary_embedding = self.embedding_model.encode_query(video_metadata.ai_summary)
            
            ids.append(summary_id)
            video_ids.append(video_metadata.video_id)
            video_urls.append(video_metadata.video_url)
            titles.append(video_metadata.title)
            channel_names.append(video_metadata.channel_name)
            languages.append(video_metadata.language.value)
            ai_summaries.append(video_metadata.ai_summary)
            keywords_list.append(" ".join(video_metadata.keywords))
            categories.append(video_metadata.category or "")
            duration_seconds_list.append(video_metadata.duration_seconds or 0)
            content_types.append("summary")
            chunk_start_times.append(0)
            chunk_end_times.append(video_metadata.duration_seconds or 0)
            chunk_texts.append(video_metadata.ai_summary)
            embeddings.append(summary_embedding.tolist())
            
            # 2. 
            keywords_id = f"{video_metadata.video_id}_keywords"
            keywords_text = " ".join(video_metadata.keywords)
            keywords_embedding = self.embedding_model.encode_query(keywords_text)
            
            ids.append(keywords_id)
            video_ids.append(video_metadata.video_id)
            video_urls.append(video_metadata.video_url)
            titles.append(video_metadata.title)
            channel_names.append(video_metadata.channel_name)
            languages.append(video_metadata.language.value)
            ai_summaries.append(video_metadata.ai_summary)
            keywords_list.append(keywords_text)
            categories.append(video_metadata.category or "")
            duration_seconds_list.append(video_metadata.duration_seconds or 0)
            content_types.append("keywords")
            chunk_start_times.append(0)
            chunk_end_times.append(video_metadata.duration_seconds or 0)
            chunk_texts.append(keywords_text)
            embeddings.append(keywords_embedding.tolist())
            
            # 3. 
            for i, chunk in enumerate(video_metadata.semantic_chunks):
                chunk_id = f"{video_metadata.video_id}_chunk_{i}"
                chunk_embedding = self.embedding_model.encode_query(chunk.text)
                
                ids.append(chunk_id)
                video_ids.append(video_metadata.video_id)
                video_urls.append(video_metadata.video_url)
                titles.append(video_metadata.title)
                channel_names.append(video_metadata.channel_name)
                languages.append(video_metadata.language.value)
                ai_summaries.append(video_metadata.ai_summary)
                keywords_list.append(" ".join(video_metadata.keywords))
                categories.append(video_metadata.category or "")
                duration_seconds_list.append(video_metadata.duration_seconds or 0)
                content_types.append("chunk")
                chunk_start_times.append(chunk.start_time_seconds)
                chunk_end_times.append(chunk.end_time_seconds)
                chunk_texts.append(chunk.text)
                embeddings.append(chunk_embedding.tolist())
            
            # 
            entities = [
                ids, video_ids, video_urls, titles, channel_names, languages,
                ai_summaries, keywords_list, categories, duration_seconds_list,
                content_types, chunk_start_times, chunk_end_times, chunk_texts, embeddings
            ]
            
            self.collection.insert(entities)
            self.collection.flush()
            
            # 
            try:
                load_state = utility.load_state(self.collection_name)
                if hasattr(load_state, 'state'):
                    is_loaded = load_state.state.name == "Loaded"
                else:
                    is_loaded = str(load_state) == "Loaded"
                
                if not is_loaded:
                    print(f" ...")
                    self.collection.load()
            except Exception as e:
                print(f"  : {e}")
                self.collection.load()
            
            total_records = len(ids)
            print(f" : {total_records} ")
            print(f"    ID: {video_metadata.video_id}")
            print(f"    : {video_metadata.title}")
            print(f"    : {len(video_metadata.semantic_chunks)} ")
            print(f"    : {total_records} ")
            
            return True
            
        except Exception as e:
            print(f" : {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_videos(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """"""
        if not self.milvus_connected:
            print(" Milvus")
            return []
        
        try:
            print(f" : {query}")
            
            # 
            try:
                load_state = utility.load_state(self.collection_name)
                if hasattr(load_state, 'state'):
                    is_loaded = load_state.state.name == "Loaded"
                else:
                    is_loaded = str(load_state) == "Loaded"
                
                if not is_loaded:
                    print(f" ...")
                    self.collection.load()
                    print(f" ")
            except Exception as e:
                print(f"  : {e}")
                self.collection.load()
                print(f" ")
            
            # 
            start_time = time.time()
            query_embedding = self.embedding_model.encode_query(query)
            embed_time = time.time() - start_time
            
            print(f" : {embed_time:.3f}")
            
            # Milvus
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            start_time = time.time()
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=[
                    "video_id", "video_url", "title", "channel_name", "language",
                    "ai_summary", "keywords", "category", "duration_seconds",
                    "content_type", "chunk_start_time", "chunk_end_time", "chunk_text"
                ]
            )
            search_time = time.time() - start_time
            
            print(f" : {search_time:.3f}")
            
            # 
            search_results = []
            
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    
                    # entity
                    def safe_get_entity_value(entity, field_name, default_value):
                        try:
                            if hasattr(entity, 'get'):
                                return entity.get(field_name) or default_value
                            elif hasattr(entity, field_name):
                                return getattr(entity, field_name) or default_value
                            else:
                                return default_value
                        except:
                            return default_value
                    
                    result = {
                        "id": hit.id,
                        "score": float(hit.score),
                        "video_id": safe_get_entity_value(entity, "video_id", ""),
                        "video_url": safe_get_entity_value(entity, "video_url", ""),
                        "title": safe_get_entity_value(entity, "title", ""),
                        "channel_name": safe_get_entity_value(entity, "channel_name", ""),
                        "language": safe_get_entity_value(entity, "language", ""),
                        "ai_summary": safe_get_entity_value(entity, "ai_summary", ""),
                        "keywords": safe_get_entity_value(entity, "keywords", ""),
                        "category": safe_get_entity_value(entity, "category", ""),
                        "duration_seconds": safe_get_entity_value(entity, "duration_seconds", 0),
                        "content_type": safe_get_entity_value(entity, "content_type", ""),
                        "chunk_start_time": safe_get_entity_value(entity, "chunk_start_time", 0),
                        "chunk_end_time": safe_get_entity_value(entity, "chunk_end_time", 0),
                        "chunk_text": safe_get_entity_value(entity, "chunk_text", "")
                    }
                    
                    search_results.append(result)
            
            print(f"  {len(search_results)} ")
            
            if search_results:
                scores = [r["score"] for r in search_results]
                print(f"   : {max(scores):.4f}")
                print(f"   : {np.mean(scores):.4f}")
            
            return search_results
            
        except Exception as e:
            print(f" : {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def generate_video_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """"""
        if not search_results:
            return ""
        
        try:
            print(" ...")
            
            # 
            video_context = self._build_video_context(search_results)
            
            # 
            prompt = self._build_qwen_prompt(query, video_context)
            
            print(f" : {len(prompt)} ")
            
            # Qwen
            start_time = time.time()
            
            response = dashscope.Generation.call(
                model="qwen-plus",
                prompt=prompt,
                max_tokens=1500,
                temperature=0.7,
                top_p=0.8
            )
            
            generate_time = time.time() - start_time
            
            if response.status_code == 200:
                answer = response.output.text.strip()
                print(f" : {generate_time:.2f}")
                return answer
            else:
                print(f" Qwen: {response}")
                return ""
                
        except Exception as e:
            print(f" : {e}")
            return f": {str(e)}"
    
    def _build_video_context(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """"""
        # ID
        videos = {}
        
        for result in search_results:
            video_id = result["video_id"]
            if video_id not in videos:
                videos[video_id] = {
                    "video_id": video_id,
                    "video_url": result["video_url"],
                    "title": result["title"],
                    "channel_name": result["channel_name"],
                    "ai_summary": result["ai_summary"],
                    "keywords": result["keywords"],
                    "category": result["category"],
                    "duration_seconds": result["duration_seconds"],
                    "semantic_chunks": []
                }
            
            # chunks
            if result["content_type"] == "chunk":
                chunk_info = {
                    "start_time_seconds": result["chunk_start_time"],
                    "end_time_seconds": result["chunk_end_time"],
                    "text": result["chunk_text"],
                    "similarity_score": result["score"]
                }
                videos[video_id]["semantic_chunks"].append(chunk_info)
        
        # 
        if videos:
            # 
            for video_id, video_data in videos.items():
                if video_data["semantic_chunks"]:
                    avg_score = np.mean([chunk["similarity_score"] for chunk in video_data["semantic_chunks"]])
                    video_data["avg_similarity"] = avg_score
                else:
                    # chunkssummary
                    summary_results = [r for r in search_results if r["video_id"] == video_id and r["content_type"] == "summary"]
                    if summary_results:
                        video_data["avg_similarity"] = summary_results[0]["score"]
                    else:
                        video_data["avg_similarity"] = 0.0
            
            # 
            best_video = max(videos.values(), key=lambda x: x["avg_similarity"])
            return best_video
        
        return {}
    
    def _build_qwen_prompt(self, query: str, video_context: Dict[str, Any]) -> str:
        """Qwen"""
        
        # 
        semantic_chunks_text = ""
        if video_context.get("semantic_chunks"):
            for i, chunk in enumerate(video_context["semantic_chunks"][:3], 1):  # 3
                start_min = chunk["start_time_seconds"] // 60
                start_sec = chunk["start_time_seconds"] % 60
                end_min = chunk["end_time_seconds"] // 60
                end_sec = chunk["end_time_seconds"] % 60
                
                semantic_chunks_text += f"""
    {{
      "": "{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}",
      "": {chunk["similarity_score"]:.4f},
      "": "{chunk["text"]}"
    }},"""
            
            semantic_chunks_text = semantic_chunks_text.rstrip(",")
        
        prompt = f""" (Role):
AI

 (Context):
"{video_context.get('title', '')}"

[]
{{
  "video_url": "{video_context.get('video_url', '')}",
  "title": "{video_context.get('title', '')}",
  "channel_name": "{video_context.get('channel_name', '')}",
  "category": "{video_context.get('category', '')}",
  "ai_summary": "{video_context.get('ai_summary', '')}",
  "keywords": "{video_context.get('keywords', '')}",
  "semantic_chunks": [{semantic_chunks_text}
  ]
}}

 (Task & Instructions):


1. : []ai_summarysemantic_chunks

2. : 

3. : video_url""

4. : ""

 (User's Query):
"{query}"

"""

        return prompt
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """RAG"""
        start_time = time.time()
        
        print(f"\n RAG")
        print(f" : {question}")
        print("="*80)
        
        # 
        search_results = self.search_videos(question, top_k)
        
        if not search_results:
            return {
                "question": question,
                "answer": "",
                "search_results": [],
                "total_time": time.time() - start_time
            }
        
        # 
        answer = self.generate_video_answer(question, search_results)
        
        total_time = time.time() - start_time
        
        # 
        result = {
            "question": question,
            "answer": answer,
            "search_results": search_results,
            "total_time": total_time,
            "stats": {
                "retrieved_videos": len(set([r["video_id"] for r in search_results])),
                "retrieved_chunks": len(search_results),
                "avg_similarity": np.mean([r["score"] for r in search_results]),
                "max_similarity": max([r["score"] for r in search_results])
            }
        }
        
        # 
        print("\n" + "="*80)
        print(" RAG")
        print("="*80)
        print(f" : {question}")
        print(f" : {answer}")
        print(f"\n :")
        print(f"   ⏱  : {total_time:.2f}")
        print(f"    : {result['stats']['retrieved_videos']}")
        print(f"    : {result['stats']['retrieved_chunks']}")
        print(f"    : {result['stats']['avg_similarity']:.4f}")
        print(f"    : {result['stats']['max_similarity']:.4f}")
        
        return result


if __name__ == "__main__":
    # RAG
    from ragflow_integration.video_metadata_schema import EXAMPLE_DIABETES_VIDEO
    
    print(" RAG")
    print("=" * 50)
    
    # 
    pipeline = VideoRAGPipeline()
    
    # 
    print("\n ...")
    success = pipeline.store_video_metadata(EXAMPLE_DIABETES_VIDEO)
    
    if success:
        # 
        test_queries = [
            "2",
            "",
            "",
            ""
        ]
        
        for query in test_queries:
            print(f"\n : {query}")
            result = pipeline.query(query, top_k=3)
            print(f"    : {len(result['search_results'])} ")
    else:
        print(" ") 