# RAG-Searx - Multi-Path Recall RAG System

A comprehensive RAG (Retrieval-Augmented Generation) system with multi-path recall capabilities for enhanced document retrieval and intelligent Q&A.

## Features

### Core Technologies
- **Multi-Path Recall**: Vector recall, keyword recall, semantic expansion, and intelligent fusion
- **BGE Embedding**: BAAI/bge-large-zh-v1.5 for high-quality Chinese text embeddings
- **RAGFlow Integration**: Advanced RAG pipeline with video content processing
- **Qwen LLM**: Integration with Alibaba's Qwen-Plus for intelligent generation

### Multi-Path Recall System
1. **Vector Recall** - Semantic similarity matching using BGE embeddings
2. **Keyword Recall** - Precise BM25-based lexical matching
3. **Semantic Expansion** - Synonym replacement and query expansion
4. **Intelligent Fusion** - Weighted fusion and re-ranking of multiple paths

### Video RAG Capabilities
- YouTube video content processing
- Semantic chunking with timestamps
- AI-generated summaries and keywords
- Time-based content retrieval

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your API keys
```

## Quick Start

### Multi-Path Recall Demo
```bash
# Interactive multi-path recall Q&A
./scripts/run_video_rag.sh multi_interactive

# Compare single vs multi-path recall
./scripts/run_video_rag.sh multi_comparison

# Full multi-path recall demo
./scripts/run_video_rag.sh multi_recall
```

### Traditional RAG Demo
```bash
# Interactive Q&A mode
./scripts/run_video_rag.sh interactive

# Full video RAG demo
./scripts/run_video_rag.sh full
```

## Architecture

### Multi-Path Recall Pipeline
```
Query → [Vector Recall]    → Semantic similarity matching
      → [Keyword Recall]   → BM25 lexical matching
      → [Semantic Expand]  → Synonym and query expansion
      → [Intelligent Fusion] → Weighted ranking and fusion
```

### Core Components
- `ragflow_integration/` - Core RAG system integration
- `demos/` - Demo scripts and examples
- `scripts/` - Execution scripts
- `src/` - Source code modules

## Configuration

### Multi-Recall Configuration
```python
MultiRecallConfig(
    vector_weight=0.4,      # Vector recall weight
    keyword_weight=0.4,     # Keyword recall weight
    semantic_weight=0.2,    # Semantic expansion weight
    final_top_k=10          # Final result count
)
```

### Video Metadata Schema
```python
VideoMetadata(
    video_id="example_id",
    video_url="https://youtube.com/watch?v=...",
    title="Video Title",
    ai_summary="AI-generated summary",
    keywords=["keyword1", "keyword2"],
    semantic_chunks=[SemanticChunk(...)]
)
```

## Usage Examples

### Python API
```python
from ragflow_integration.enhanced_video_rag_pipeline import EnhancedVideoRAGPipeline
from ragflow_integration.multi_path_recall import MultiRecallConfig

# Configure multi-path recall
config = MultiRecallConfig(
    vector_weight=0.4,
    keyword_weight=0.4,
    semantic_weight=0.2
)

# Initialize pipeline
pipeline = EnhancedVideoRAGPipeline(recall_config=config)

# Store video metadata
pipeline.store_video_metadata(video_metadata)

# Query with multi-path recall
result = pipeline.enhanced_query("Your question here")
print(result["answer"])
```

### Command Line
```bash
# Multi-path recall modes
./scripts/run_video_rag.sh multi_interactive
./scripts/run_video_rag.sh multi_comparison
./scripts/run_video_rag.sh multi_recall

# Traditional modes
./scripts/run_video_rag.sh qa
./scripts/run_video_rag.sh interactive
```

## Docker Deployment

```bash
# Build and run with Docker
docker-compose up -d

# Access the API
curl http://localhost:8000/docs
```

## Directory Structure

```
rag-searx/
├── ragflow_integration/     # Core RAG system
│   ├── multi_path_recall.py
│   ├── enhanced_video_rag_pipeline.py
│   ├── video_metadata_schema.py
│   └── bge_embedding.py
├── demos/                   # Demo scripts
│   ├── multi_recall_demo.py
│   ├── video_rag_demo.py
│   └── simple_video_rag_demo.py
├── scripts/                 # Execution scripts
│   └── run_video_rag.sh
├── src/                     # Source modules
├── config/                  # Configuration files
├── docs/                    # Documentation
└── requirements.txt         # Dependencies
```

## API Documentation

When running with FastAPI, visit `http://localhost:8000/docs` for interactive API documentation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- BGE Embedding Model by BAAI
- Qwen LLM by Alibaba Cloud
- RAGFlow Framework
- Milvus Vector Database
