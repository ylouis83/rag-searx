#  RAGFlow

##  

RAGFlowYouTube

###  

YouTubeAI

###  

```
 → RAGFlow→ Qwen
```

##  

### 

- **RAGFlow**: 
- **BGE-large-zh-v1.5**:   
- **Milvus**: 
- **Qwen-Plus**: 
- ****: Prompt

##  Schema

### 

```json
{
  "video_id": "au-w0QXB6jg",
  "video_url": "https://www.youtube.com/watch?v=au-w0QXB6jg", 
  "title": " 2  (Understanding Type 2 Diabetes Mellitus)",
  "channel_name": "Nucleus Medical Media",
  "language": "zh-CN",
  "category": "",
  
  "ai_summary": "2...",
  
  "keywords": [
    "2", "", "", ""
  ],
  
  "semantic_chunks": [
    {
      "start_time_seconds": 46,
      "end_time_seconds": 75,
      "text": "2''..."
    }
  ]
}
```

### 


-  **AI** - 
-  **** - 
-  **** - 
-  **** - 

##  

### 

#### 1.  (Role)
```
AI

```

#### 2.  (Context)
```json
[]
{
  "video_url": "...",
  "title": "...",
  "ai_summary": "...",
  "semantic_chunks": [...]
}
```

#### 3.  (Instructions)
- ****: 
- ****: 
- ****: 
- ****: 

#### 4.  (Query)
- 
- 

### 

-  **** - JSON
-  **** - AI++
-  **** - 
-  **** - 4
-  **** - 
-  **** - 

##  

```
rag-searx/
 ragflow_integration/          # RAG
    video_metadata_schema.py  # Schema
    video_rag_pipeline.py     # RAG
    bge_embedding.py          # BGE

 demos/                        # 
    video_rag_demo.py         # RAG
    simple_video_rag_demo.py  # Milvus

 scripts/                      # 
    run_video_rag.sh          # RAG

 docs/                         # 
    RAG.md        # 

 run_video_rag.sh              # 
```

##  

### 1. 

```bash
# 
source venv/bin/activate

# 
pip install -r requirements.txt
```

### 2. 

#### Milvus
```bash
python demos/simple_video_rag_demo.py
```

#### Milvus
```bash
./run_video_rag.sh full
```

### 3. 

```bash
# 
python demos/simple_video_rag_demo.py metadata    # 
python demos/simple_video_rag_demo.py prompt      #   
python demos/simple_video_rag_demo.py qa          # 
python demos/simple_video_rag_demo.py interactive # 

# Milvus
./run_video_rag.sh full        # 
./run_video_rag.sh schema      # Schema
./run_video_rag.sh qa          # 
./run_video_rag.sh interactive # 
```

##  

### 1. 

-  **YouTube**: IDURL
-  **AI**: 
-  ****: 
-  ****: 
-  ****: 

### 2. 

-  ****: BGE-large-zh-v1.5
-  ****: Milvus
-  ****: COSINE
-  ****: 

### 3. 

-  **Qwen**: Plus
-  **Prompt**: 
-  ****: 
-  ****: 

##  

### 
-  ****: 
-  ****: 
-  ****: 
-  ****: 

### 
-  ****: 
-  ****: 
-  ****: 
-  ****: 

##  

### 
```
2
```

### 
```
2""



• 
• 
• 


• 
• 
• 


https://www.youtube.com/watch?v=au-w0QXB6jg


```

##  

### 1. 
-  **Schema**: 
- ⏰ ****: 
-  ****: 
-  ****: 

### 2. 
-  ****: 
-  ****: 
-  ****: 4
-  ****: 

### 3. 
-  ****: 
-  ****: 
-  ****: 
-  ****: +

##  

### 1. 


### 2. 


### 3. 
Milvus

### 4. 


### 5. 


---

##  


- : Mr.Louis
- : RAGFlow + BGE + Milvus + Qwen + 

** ** 