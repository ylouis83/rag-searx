# RAG-SEARX 

##  

```
rag-searx/
  apps/                    # 
    simple_app.py          # RAG
    enhanced_app.py        # RAG

  demos/                   # 
    ragflow_bge_demo.py    # RAGFlow + BGE
    enhanced_rag_demo.py   # RAG
    enhanced_rag_demo_markdown.py  # Markdown RAG
    simple_vector_demo.py  # 
    demo_vector.py         # 

  tests/                   # 
    test_rag_pipeline.py   # RAG
    pdf_rag_test.py        # PDF RAG
    large_vector_test.py   # 
    vector_test.py         # 
    test_milvus.py         # Milvus

  tools/                   # 
    check_collection_index.py    # 
    inspect_pdf_collection.py    # PDF
    check_milvus_collections.py  # Milvus

  scripts/                 # 
    run_ragflow_bge.sh     # RAGFlow + BGE
    run_markdown_demo.sh   # Markdown
    run_demo.sh            # 
    quick_demo.sh          # 
    start.sh               # 
    stop.sh                # 
    build_index.py         # 

  test_data/               # 
    ragtest.pdf            # RAGPDF
    large_test_document.txt # 
    test_document.txt      # 
    test_book.txt          # 

  docs/                    # 
    RAGFlow_BGE_.md     # RAGFlow + BGE
    Enhanced_RAG_Demo_.md   # RAG

  logs/                    # 
    app.log                # 
    rag_searx.log          # RAG

  ragflow_integration/     # RAGFlow
    __init__.py
    ragflow_rag_pipeline.py      # RAGFlow RAG
    document_processor.py        # 
    bge_embedding.py             # BGE
    chunk_models.py              # 

  config/                  # 
  frontend/                # 
  src/                     # 
  data/                    # 
  uploads/                 # 
  volumes/                 # Docker
  venv/                    # Python

  requirements.txt         # 
  requirements_minimal.txt # 
  Dockerfile              # Docker
  docker-compose.yml      # Docker Compose
  env.example              # 
  README.md               # 
```

##  

### 1.  (`apps/`)
- **simple_app.py**: RAG
- **enhanced_app.py**: RAGWeb

### 2.  (`demos/`)
- **ragflow_bge_demo.py**: RAGFlow + BGE
- **enhanced_rag_demo.py**: RAG
- **vector_demo**: 

### 3.  (`ragflow_integration/`)
- **ragflow_rag_pipeline.py**: RAG
- **document_processor.py**: 
- **bge_embedding.py**: BGE

##  

### 
```bash
# RAGFlow + BGE
./scripts/run_ragflow_bge.sh full

# 
./scripts/run_ragflow_bge.sh process

# 
./scripts/run_ragflow_bge.sh qa

# 
./scripts/run_ragflow_bge.sh interactive
```

### 
```bash
# 
python apps/simple_app.py

# 
python apps/enhanced_app.py
```

### 
```bash
# RAG
python tests/test_rag_pipeline.py

# PDF
python tests/pdf_rag_test.py
```

##  

###  (`tools/`)
- Milvus
- 
- PDF

###  (`test_data/`)
- 
- 

##  

- `docs/RAGFlow_BGE_.md`: 
- `docs/Enhanced_RAG_Demo_.md`: 
- `README.md`: 

##  

-  **RAGFlow**: 
-  **BGE**: BGE-large-zh-v1.5
-  **Milvus**: 
-  **Qwen**: Qwen-Plus
-  ****: demo

##  

- `requirements.txt`: 
- `requirements_minimal.txt`: 
- `venv/`: Python

##  

- `Dockerfile`: 
- `docker-compose.yml`: 
- `volumes/`: 

---

*: 2025-06-23*
*: AI Assistant* 