#  RAG

##  

**macOS iTerm2**RAGMilvusQwen3

##  

###  
- **** - ANSIiTerm2
- **** - 
- **** - 
- **** - ℹ

###  

#### 1. 
- Milvus
- 
- 
- 

#### 2. 
- 
- 
- 
- 

#### 3. 
- 
- 
- 
- 

#### 4. 
- ****ID
- ****
- ****
- ****

#### 5. LLM
- 
- 
- 
- 

#### 6. 
- ****
- **AI**
- ****
- ****

##  

### 
```bash
# 
python enhanced_rag_demo.py

# ""
```

### 
```bash
# 
python enhanced_rag_demo.py ""
python enhanced_rag_demo.py ""
python enhanced_rag_demo.py ""
```

### 
```bash
# 
chmod +x enhanced_rag_demo.py
./enhanced_rag_demo.py ""
```

##  

### 

```

                         RAG                            
  : Milvus + Qwen3                                   
  : RAG (→→)                                    
  : macOS iTerm2                                                    


 Milvus...
 Milvus
ℹ  : localhost:19530
 : pdf_rag_test
ℹ    10 

 :
ℹ  : embedding
ℹ  : IVF_FLAT
ℹ  : IP
ℹ  : 6

 RAG
================================================================================

 : 
  Milvus...
  ...
 
  5 

 :

  1
   ID: xxx...
   : ragtest.pdf
   : X
   : X
   : X.XXXX
   : ...

  Qwen3...
 Qwen3

================================================================================
 RAG - 
================================================================================

 : 

 AI:

[AI]


 :
  ⏱  : XX.XX
   : X
   : X.XXXX
   : X.XXXX
   : XXX

 :
  1. ragtest.pdf (X) - : X.XXXX
  2. ragtest.pdf (X) - : X.XXXX
  3. ragtest.pdf (X) - : X.XXXX

================================================================================
  RAG
```

##  

### 
- **Milvus** - 
- **DashScope** - LLM
- **Python** - 
- **ANSI** - 

### 
- **** - IPTop-K
- **** - 
- **** - 
- **** - 

##  

### Python
```txt
pymilvus>=2.3.4
dashscope>=1.14.1
numpy>=1.24.4
```

### 
- ****: macOS (iTerm2)
- **Python**: 3.9+
- **Milvus**: 2.3+ (localhost:19530)
- ****: DashScope API

##  

### ragtest.pdf
- ****: 10
- ****: 1536
- ****: ~0.4/
- ****: ~0.03/
- **LLM**: ~45/
- ****: 18-35

### 
- ****: 3000-6000
- ****: 1800-5400
- ****: Top-5100%

##  

### 
```bash
# 
Ctrl + C  # 

# 
python enhanced_rag_demo.py -h

# 
for query in "AI" "" ""; do
    python enhanced_rag_demo.py "$query"
done
```

### 
- **** - 
- **API** - 
- **** - 
- **** - Ctrl+C

##  

### 
1. **** - 
2. **** - top_ktemperature
3. **** - JSON/Markdown
4. **** - 

### 
1. **** - 
2. **** - 
3. **** - LLM
4. **** - 

##  

RAGRAG


- **RAG** - 
- **** - 
- **** - 
- **** - 

macOS iTerm2RAG

# Enhanced RAG Demo 

## 

MilvusDashScope APIRAG (Retrieval-Augmented Generation) 

## 

1. **enhanced_rag_demo.py** - (ANSI)
2. **enhanced_rag_demo_markdown.py** - Markdown   ****
3. **enhanced_app.py** - FastAPI Web
4. **quick_demo.sh** - 
5. **run_demo.sh** - 
6. **run_markdown_demo.sh** - Markdown   ****

## 

### 1: Markdown ()

```bash
# 
./run_markdown_demo.sh

# 
./run_markdown_demo.sh ""
```

### 2:  ()

```bash
# 
./run_demo.sh

# 
./run_demo.sh ""
```

### 3: 

```bash
./quick_demo.sh
```

### 4: Web

```bash
# Web
python enhanced_app.py

#  http://localhost:8000
```

## 

###  (enhanced_rag_demo.py)
- ****: ANSI
- ****: 
- ****: 

### Markdown (enhanced_rag_demo_markdown.py)   ****
- ****: Markdown
- ****: 
- ****: Markdown
- ****: 
- ****: 

## 

###  
1. ****: DashScope text-embedding-v1
2. ****: MilvusTop-K
3. ****: 
4. ****: 

###  
1. ****: 
2. ****: LLM
3. **Qwen3**: qwen-plus
4. ****: 

###  
- 
- 
- 
- 
- 
- 

## 

 `ragtest.pdf`AI
- ****: 200KB
- ****: 3
- ****: 10
- ****: 

## 


1. `""`
2. `""`
3. `"AlexNet"`

##  (Markdown)

```markdown
#  RAG (Markdown)

##  
 **Milvus**
- : `localhost:19530`

###  

#### 
|  |  |
|------|------|
|  | 5 |
|  | 3368.1116 |
|  | 1839.4399 |

###  
```
[1 - : ragtest.pdf 1]
......
```

## 

### 
- ****: 1536
- ****: ~0.4/
- ****: ~0.02/
- ****: Top-5100%

### 
- **LLM**: 40-50/
- ****: 18-25
- ****: 

## 

### 
- Python 3.9+
- Milvus 2.3.4+
- DashScope API

### 
```bash
pip install pymilvus==2.3.4
pip install dashscope==1.14.1
pip install fastapi==0.104.1
pip install uvicorn
```

## API

```python
# DashScope API
DASHSCOPE_API_KEY = "sk-b70842d25c884aa9aa18955b00c24d37"

# 
EMBEDDING_MODEL = "text-embedding-v1"
LLM_MODEL = "qwen-plus"
```

## 

### 
1. **ModuleNotFoundError**: 
2. **Milvus**: Milvus
3. **API**: DashScope API

### 
```bash
# rag-searx
source venv/bin/activate
```

## 

### v2.0.0 ()  
- ****: Markdown
- ****: 
- ****: 
- ****: 

### v1.0.0
- RAG
- ANSI

---

## 

**AI-Coding R&D Team**  
RAG 