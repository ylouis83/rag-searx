# RAGFlow + BGE 

##  

 **RAGFlow**  **BGE-large-zh-v1.5** RAG

##  

###  vs 

|  |  |  (RAGFlow + BGE) |
|------|--------|------------------------|
| **** |  | RAGFlow |
| **** | DashScope text-embedding-v1 (1536) | BGE-large-zh-v1.5 (1024) |
| **** |  | (///) |
| **** |  |  |
| **** |  |  |

### 

- ** **: RAGFlow
- ** **: BGE-large-zh-v1.5 (BAAI/Beijing Academy of AI)
- ** **: Milvus ()
- ** **: Qwen-Plus (DashScope)
- ** **: Python 3.9+

##  

```
rag-searx/
 ragflow_integration/           # RAGFlow + BGE
    __init__.py               # 
    document_processor.py     # RAGFlow
    bge_embedding.py          # BGE
    ragflow_rag_pipeline.py   # RAG
 ragflow_bge_demo.py           # 
 run_ragflow_bge.sh            # 
 RAGFlow_BGE_.md   # 
```

##  

### 1. RAGFlow (`document_processor.py`)

#### 

|  |  |  |
|------|----------|------|
| **BOOK** |  |  |
| **PAPER** |  | AbstractIntroduction |
| **QA** |  |  |
| **TABLE** |  |  |
| **NAIVE** |  |  |

#### 

- ** **: 
- ** Token**: chunk
- ** **: 
- ** **: 
- ** **: 

### 2. BGE (`bge_embedding.py`)

#### 

- ****: BAAI/bge-large-zh-v1.5
- ****: 1024
- ****: 
- ****: (CUDA/MPS/CPU)
- ****: L2

#### 

```python
#  ()
doc_embedding = model.encode_document("")

#  ()
query_embedding = model.encode_query("")

# 
embeddings = model.batch_encode(texts, batch_size=32)

# 
similarity = model.compute_similarity(emb1, emb2)
```

### 3. RAG (`ragflow_rag_pipeline.py`)

#### 

1. ** **
   - RAGFlow
   - 
   - 

2. ** **
   - BGE
   - 
   - 

3. ** **
   - Milvus
   - 
   - 

4. ** **
   - 
   - 
   - 

5. ** **
   - 
   - Qwen-Plus
   - 

##  Milvus

### 

|  |  |  |  |
|--------|------|------|------|
| `id` | VARCHAR | 255 |  |
| `content` | VARCHAR | 10000 |  |
| `file_name` | VARCHAR | 255 |  |
| `page_number` | INT64 | - |  |
| `chunk_index` | INT64 | - |  |
| `chapter_title` | VARCHAR | 500 |  |
| `keywords` | VARCHAR | 1000 |  |
| `embedding` | FLOAT_VECTOR | 1024 | BGE |

### 

- ****: COSINE ()
- ****: IVF_FLAT
- ****: nlist=1024, nprobe=10

##  

### 1. 

```bash
# 
./run_ragflow_bge.sh

# 
./run_ragflow_bge.sh interactive  # 
./run_ragflow_bge.sh process      # 
./run_ragflow_bge.sh embed        # 
```

### 2. 

```python
from ragflow_integration import RAGFlowPipeline, ChunkConfig, ChunkingStrategy

# 
config = ChunkConfig(
    strategy=ChunkingStrategy.BOOK,
    chunk_token_count=256,
    chunk_overlap=0.1,
    auto_keywords=True,
    auto_question=True
)

# 
pipeline = RAGFlowPipeline(
    chunk_config=config,
    embedding_model_name="BAAI/bge-large-zh-v1.5",
    collection_name="my_knowledge_base"
)

# 
pipeline.process_and_store_document("document.pdf")

# 
result = pipeline.query("", top_k=5)
print(result["answer"])
```

##  

### 1. 

- ****: RAGFlow
- ****: 
- ****: 

### 2. 

- ****: BGE-large-zh-v1.5
- ****: 1024
- ****: 

### 3. 

- ****: 
- ****: chunk
- ****: 

##  

### 

1. ****: Milvus
2. **API**: 
3. ****: 

### 

```python
# 
old_config = {
    "chunk_size": 400,
    "overlap": 50,
    "embedding_model": "text-embedding-v1"
}

# 
new_config = ChunkConfig(
    strategy=ChunkingStrategy.BOOK,
    chunk_token_count=256,  # token
    chunk_overlap=0.1,      # 
    auto_keywords=True,     # 
    auto_question=True      # 
)
```

##  

### 

1. ****: 
2. ****: 
3. ****: 
4. ****: 
5. ****: 

### 

- **Web**: FastAPI
- **API**: RESTful API
- ****: 
- ****: RAG

##  

### 

1. ****: BGE
2. **Milvus**: Milvus
3. ****: BGE16GB+
4. ****: Python

### 

- **batch_size**: GPU/
- **chunk_token_count**: chunk
- **top_k**: 
- **temperature**: 

---

** RAGFlow + BGE** 