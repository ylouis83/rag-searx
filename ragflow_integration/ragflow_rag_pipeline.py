#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow + BGE RAG
RAGFlowBGE-large-zh-v1.5
"""

import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import dashscope

# Milvus
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 
from .document_processor import RAGFlowDocumentProcessor, ChunkConfig, ChunkingStrategy
from .bge_embedding import get_bge_model


class RAGFlowPipeline:
    """RAGFlow + BGE RAG"""
    
    def __init__(
        self,
        chunk_config: ChunkConfig = None,
        embedding_model_name: str = "BAAI/bge-large-zh-v1.5",
        collection_name: str = "ragflow_bge_collection"
    ):
        """RAGFlow RAG"""
        
        # 
        self.chunk_config = chunk_config or ChunkConfig(
            strategy=ChunkingStrategy.BOOK,
            chunk_token_count=256,
            chunk_overlap=0.1,
            auto_keywords=True,
            auto_question=True
        )
        
        self.collection_name = collection_name
        
        # 
        print(" RAGFlow + BGE RAG")
        
        # 
        self.doc_processor = RAGFlowDocumentProcessor(self.chunk_config)
        print(" RAGFlow")
        
        # BGE
        self.embedding_model = get_bge_model(model_name=embedding_model_name)
        print(" BGE")
        
        # LLM
        self.llm_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        os.environ["DASHSCOPE_API_KEY"] = self.llm_api_key
        dashscope.api_key = self.llm_api_key
        
        # Milvus
        self.milvus_connected = False
        self.collection = None
        
        # Milvus
        self._connect_milvus()
    
    def _connect_milvus(self):
        """Milvus"""
        
        try:
            print(f" Milvus")
            
            connections.connect(
                alias="default",
                host="localhost",
                port="19530"
            )
            
            self.milvus_connected = True
            print(" Milvus")
            
            # 
            self._ensure_collection()
            
        except Exception as e:
            print(f" Milvus: {e}")
            self.milvus_connected = False
    
    def _ensure_collection(self):
        """Milvus"""
        
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
                    # Milvus API
                    if hasattr(load_state, 'state'):
                        is_loaded = load_state.state.name == "Loaded"
                    else:
                        # Milvus
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
            
            # BGE-large-zh-v1.51024
            embedding_dim = 1024
            
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="chapter_title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
            ]
            
            schema = CollectionSchema(
                fields=fields, 
                description="RAGFlow + BGE"
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
    
    def process_and_store_document(self, file_path: str) -> bool:
        """"""
        
        if not self.milvus_connected:
            print(" Milvus")
            return False
        
        try:
            file_path = Path(file_path)
            print(f" : {file_path.name}")
            
            # 1. RAGFlow
            start_time = time.time()
            chunks = self.doc_processor.process_document(str(file_path))
            process_time = time.time() - start_time
            
            if not chunks:
                print(" ")
                return False
            
            print(f" RAGFlow: {len(chunks)} : {process_time:.2f}")
            
            # 2. BGE
            print(" BGE...")
            
            texts = [chunk.content for chunk in chunks]
            
            start_time = time.time()
            embeddings = self.embedding_model.batch_encode(texts, show_progress=True)
            embed_time = time.time() - start_time
            
            print(f" BGE: {embed_time:.2f}")
            
            # 3. Milvus
            print(" Milvus...")
            
            ids = []
            contents = []
            file_names = []
            page_numbers = []
            chunk_indices = []
            chapter_titles = []
            keywords_list = []
            embedding_list = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                ids.append(chunk.chunk_id)
                contents.append(chunk.content)
                file_names.append(file_path.name)
                page_numbers.append(chunk.page_number or 0)
                chunk_indices.append(chunk.metadata.get("chunk_index", i))
                chapter_titles.append(chunk.chapter_title or "")
                keywords_list.append(", ".join(chunk.keywords))
                embedding_list.append(embedding.tolist())
            
            entities = [
                ids, contents, file_names, page_numbers, 
                chunk_indices, chapter_titles, keywords_list, embedding_list
            ]
            
            start_time = time.time()
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
            
            store_time = time.time() - start_time
            
            print(f" : {len(chunks)} : {store_time:.2f}")
            
            return True
            
        except Exception as e:
            print(f" : {e}")
            return False
    
    def process_and_store_document_with_details(self, file_path: str) -> bool:
        """ - chunkembedding"""
        
        if not self.milvus_connected:
            print(" Milvus")
            return False
        
        try:
            file_path = Path(file_path)
            print(f" : {file_path.name}")
            print(f" : {file_path.stat().st_size / 1024:.2f} KB")
            
            # 1. RAGFlow
            print("\n" + "="*60)
            print(" RAGFlow")
            print("="*60)
            
            start_time = time.time()
            chunks = self.doc_processor.process_document(str(file_path))
            process_time = time.time() - start_time
            
            if not chunks:
                print(" ")
                return False
            
            print(f" RAGFlow:")
            print(f"    : {len(chunks)}")
            print(f"   ⏱  : {process_time:.2f}")
            print(f"    : {self.chunk_config.strategy.value}")
            print(f"    token: {self.chunk_config.chunk_token_count}")
            print(f"    : {self.chunk_config.chunk_overlap}")
            
            # chunk
            print(f"\n :")
            print("-" * 80)
            
            for i, chunk in enumerate(chunks):
                print(f"\n  {i+1}/{len(chunks)}:")
                print(f"    ID: {chunk.chunk_id}")
                print(f"    : {chunk.page_number}")
                print(f"    : {chunk.chapter_title or ''}")
                print(f"    : {len(chunk.content)} ")
                print(f"     : {chunk.keywords}")
                print(f"    : {chunk.questions}")
                print(f"    : {chunk.content[:200]}...")
                
                if i >= 2:  # 3
                    remaining = len(chunks) - 3
                    if remaining > 0:
                        print(f"\n   ... ( {remaining} )")
                    break
            
            # 2. BGE
            print(f"\n" + "="*60)
            print(" BGE")
            print("="*60)
            
            texts = [chunk.content for chunk in chunks]
            
            print(f" BGE...")
            print(f"    : {len(texts)}")
            print(f"    : 1024")
            print(f"    : BGE-large-zh-v1.5")
            
            start_time = time.time()
            embeddings = self.embedding_model.batch_encode(texts, show_progress=True)
            embed_time = time.time() - start_time
            
            print(f" BGE:")
            print(f"   ⏱  : {embed_time:.2f}")
            print(f"    : {embeddings.shape}")
            print(f"    : {embeddings.shape[1]}")
            print(f"    : {len(texts)/embed_time:.1f} /")
            
            # embedding
            print(f"\n :")
            print("-" * 80)
            
            for i, (chunk, embedding) in enumerate(zip(chunks[:3], embeddings[:3])):
                vector_norm = np.linalg.norm(embedding)
                vector_mean = np.mean(embedding)
                vector_std = np.std(embedding)
                
                print(f"\n  {i+1}:")
                print(f"    ID: {chunk.chunk_id[:16]}...")
                print(f"    : {vector_norm:.6f}")
                print(f"    : {vector_mean:.6f}")
                print(f"    : {vector_std:.6f}")
                print(f"    : [{embedding[0]:.6f}, {embedding[1]:.6f}, ..., {embedding[-1]:.6f}]")
                print(f"    : {chunk.content[:100]}...")
            
            if len(embeddings) > 3:
                print(f"\n   ... ( {len(embeddings)-3} )")
            
            # 
            if len(embeddings) >= 2:
                print(f"\n :")
                similarities = self.embedding_model.compute_similarity(
                    embeddings[0:1], embeddings
                )
                print(f"   (1):")
                for i, sim in enumerate(similarities[0][:5]):
                    print(f"      {i+1}: {sim:.6f}")
            
            # 3. Milvus
            print(f"\n" + "="*60)
            print(" Milvus")
            print("="*60)
            
            print(" Milvus...")
            
            ids = []
            contents = []
            file_names = []
            page_numbers = []
            chunk_indices = []
            chapter_titles = []
            keywords_list = []
            embedding_list = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                ids.append(chunk.chunk_id)
                contents.append(chunk.content)
                file_names.append(file_path.name)
                page_numbers.append(chunk.page_number or 0)
                chunk_indices.append(chunk.metadata.get("chunk_index", i))
                chapter_titles.append(chunk.chapter_title or "")
                keywords_list.append(", ".join(chunk.keywords))
                embedding_list.append(embedding.tolist())
            
            print(f" :")
            print(f"    ID: {len(ids)}")
            print(f"    : {len(contents)}")
            print(f"    : {file_path.name}")
            print(f"    : {len(embedding_list)}")
            print(f"    : {len(embedding_list[0]) if embedding_list else 0}")
            
            entities = [
                ids, contents, file_names, page_numbers, 
                chunk_indices, chapter_titles, keywords_list, embedding_list
            ]
            
            start_time = time.time()
            insert_result = self.collection.insert(entities)
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
                    print(f" ")
            except Exception as e:
                print(f"  : {e}")
                self.collection.load()
                print(f" ")
            
            store_time = time.time() - start_time
            
            print(f" Milvus:")
            print(f"   ⏱  : {store_time:.2f}")
            print(f"    : {len(chunks)}")
            print(f"    ID: {len(insert_result.primary_keys) if hasattr(insert_result, 'primary_keys') else len(ids)}")
            print(f"    : {self.collection_name}")
            print(f"    : ")
            
            # 4. 
            print(f"\n" + "="*60)
            print(" ")
            print("="*60)
            
            total_time = process_time + embed_time + store_time
            total_chars = sum(len(chunk.content) for chunk in chunks)
            avg_chunk_size = total_chars / len(chunks)
            
            print(f" :")
            print(f"    : {file_path.name}")
            print(f"    : {self.chunk_config.strategy.value}")
            print(f"    : {len(chunks)}")
            print(f"    : {total_chars:,}")
            print(f"    : {avg_chunk_size:.0f} ")
            print(f"    : {embeddings.shape[1]}")
            print(f"   ⏱  : {total_time:.2f}")
            print(f"       : {process_time:.2f}")
            print(f"       : {embed_time:.2f}")
            print(f"       : {store_time:.2f}")
            print(f"    : {total_chars/total_time:.0f} /")
            
            return True
            
        except Exception as e:
            print(f" : {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """"""
        
        if not self.milvus_connected:
            print(" Milvus")
            return []
        
        try:
            print(f" : {query}")
            
            # 
            try:
                load_state = utility.load_state(self.collection_name)
                # Milvus API
                if hasattr(load_state, 'state'):
                    is_loaded = load_state.state.name == "Loaded"
                else:
                    # Milvus
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
                    "content", "file_name", "page_number", 
                    "chunk_index", "chapter_title", "keywords"
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
                    
                    # 
                    content = safe_get_entity_value(entity, "content", "")
                    file_name = safe_get_entity_value(entity, "file_name", "")
                    page_number = safe_get_entity_value(entity, "page_number", 0)
                    chunk_index = safe_get_entity_value(entity, "chunk_index", 0)
                    chapter_title = safe_get_entity_value(entity, "chapter_title", "")
                    keywords = safe_get_entity_value(entity, "keywords", "")
                    
                    result = {
                        "id": hit.id,
                        "score": float(hit.score),
                        "content": content,
                        "file_name": file_name,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "chapter_title": chapter_title,
                        "keywords": str(keywords).split(", ") if keywords else []
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
            return []
    
    def generate_answer(self, query: str, context_results: List[Dict[str, Any]]) -> str:
        """"""
        
        if not context_results:
            return ""
        
        try:
            print(" ...")
            
            # 
            context_parts = []
            
            for i, result in enumerate(context_results):
                context_part = f"""[{i+1}]
: {result["file_name"]} ({result["page_number"]})
: {result["chapter_title"]}
: {', '.join(result["keywords"][:5])}
: {result["score"]:.4f}
: {result["content"]}"""
                
                context_parts.append(context_part)
            
            context = "\n\n".join(context_parts)
            
            # 
            prompt = f"""AI


{context}

{query}

"""

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
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """RAG"""
        
        start_time = time.time()
        
        print(f"\n RAGFlow + BGE")
        print(f" : {question}")
        print("="*80)
        
        # 
        search_results = self.search_documents(question, top_k)
        
        if not search_results:
            return {
                "question": question,
                "answer": "",
                "search_results": [],
                "total_time": time.time() - start_time
            }
        
        # 
        answer = self.generate_answer(question, search_results)
        
        total_time = time.time() - start_time
        
        # 
        result = {
            "question": question,
            "answer": answer,
            "search_results": search_results,
            "total_time": total_time,
            "stats": {
                "retrieved_chunks": len(search_results),
                "avg_similarity": np.mean([r["score"] for r in search_results]),
                "max_similarity": max([r["score"] for r in search_results])
            }
        }
        
        # 
        print("\n" + "="*80)
        print(" RAGFlow + BGE")
        print("="*80)
        print(f" : {question}")
        print(f" : {answer}")
        print(f"\n :")
        print(f"   ⏱  : {total_time:.2f}")
        print(f"    : {len(search_results)}")
        print(f"    : {result['stats']['avg_similarity']:.4f}")
        print(f"    : {result['stats']['max_similarity']:.4f}")
        
        return result
