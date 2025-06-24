#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow + BGE 
RAGFlowBGE-large-zh-v1.5RAG
"""

import sys
from pathlib import Path
import numpy as np

# Python
sys.path.insert(0, str(Path(__file__).parent.parent))

# RAGFlow + BGE
from ragflow_integration.ragflow_rag_pipeline import RAGFlowPipeline
from ragflow_integration.document_processor import ChunkConfig, ChunkingStrategy


def print_banner():
    """"""
    
    banner = """

                     RAGFlow + BGE                           
                                                                              
   :                                                                
     • RAGFlow  (///)                          
     • BGE-large-zh-v1.5  (1024)                        
     • Milvus  +                                     
     • Qwen-Plus                                           
                                                                              
   : Python + RAGFlow + BGE + Milvus + DashScope API                

    """
    
    print(banner)


def demo_document_processing():
    """"""
    
    print("\n" + "="*80)
    print(" RAGFlow + BGE ")
    print("="*80)
    
    # RAGFlow
    chunk_config = ChunkConfig(
        strategy=ChunkingStrategy.BOOK,     # 
        chunk_token_count=256,              # chunk 256 tokens
        chunk_overlap=0.1,                  # 10% 
        auto_keywords=True,                 # 
        auto_question=True                  # 
    )
    
    # RAGFlow
    print(" RAGFlow + BGE...")
    pipeline = RAGFlowPipeline(
        chunk_config=chunk_config,
        embedding_model_name="BAAI/bge-large-zh-v1.5",
        collection_name="ragflow_bge_demo"
    )
    
    #  - ragtest.pdf
    target_file = "test_data/ragtest.pdf"
    
    if Path(target_file).exists():
        print(f"\n : {target_file}")
        success = pipeline.process_and_store_document_with_details(target_file)
        
        if success:
            print(f" {target_file} ")
        else:
            print(f" {target_file} ")
    else:
        print(f"  : {target_file}")
        print(" ragtest.pdf ")
    
    return pipeline


def demo_intelligent_qa(pipeline: RAGFlowPipeline):
    """"""
    
    print("\n" + "="*80)
    print(" RAGFlow + BGE ")
    print("="*80)
    
    # 
    test_queries = [
        "",
        "",
        "AlexNet",
        "",
        "",
        ""
    ]
    
    print(" ...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"  {i}/{len(test_queries)}")
        print(f"{'='*60}")
        
        # RAG
        result = pipeline.query(query, top_k=3)
        
        # 
        print(f"\n :")
        for j, search_result in enumerate(result["search_results"], 1):
            print(f"\n     {j}:")
            print(f"       : {search_result['file_name']}")
            print(f"       : {search_result['page_number']}")
            print(f"       : {search_result['chapter_title']}")
            print(f"        : {', '.join(search_result['keywords'][:3])}")
            print(f"       : {search_result['score']:.4f}")
            print(f"       : {search_result['content'][:100]}...")
        
        print(f"\n :")
        merged_content = "\n\n".join([r['content'] for r in result["search_results"]])
        print(f"    : {len(merged_content)} ")
        print(f"    : {set([r['file_name'] for r in result['search_results']])}")
        
        # 
        if i < len(test_queries):
            input(f"\n⏳ Enter...")


def demo_chunking_strategies():
    """"""
    
    print("\n" + "="*80)
    print(" RAGFlow ")
    print("="*80)
    
    # 
    strategies = [
        (ChunkingStrategy.BOOK, " - "),
        (ChunkingStrategy.PAPER, " - "),
        (ChunkingStrategy.QA, " - "),
        (ChunkingStrategy.TABLE, " - ")
    ]
    
    test_text = """


Artificial IntelligenceAI

1.1 

1956








2.1 


    """
    
    from ragflow_integration.document_processor import RAGFlowDocumentProcessor
    
    for strategy, description in strategies:
        print(f"\n : {description}")
        print(f"   : {strategy.value}")
        
        # 
        config = ChunkConfig(
            strategy=strategy,
            chunk_token_count=128,
            chunk_overlap=0.1,
            auto_keywords=True,
            auto_question=True
        )
        
        # 
        processor = RAGFlowDocumentProcessor(config)
        
        try:
            # 
            chunks = processor._chunk_text_by_strategy(test_text)
            
            print(f"    : {len(chunks)}")
            
            for i, chunk in enumerate(chunks[:2], 1):  # 2
                print(f"\n     {i}:")
                print(f"       ID: {chunk.chunk_id[:16]}...")
                print(f"       : {chunk.content[:150]}...")
                print(f"        : {chunk.keywords}")
                print(f"       : {chunk.questions}")
                
        except Exception as e:
            print(f"    : {e}")


def demo_bge_embedding():
    """BGE"""
    
    print("\n" + "="*80)
    print(" BGE-large-zh-v1.5 ")
    print("="*80)
    
    from ragflow_integration.bge_embedding import get_bge_model
    
    # BGE
    bge_model = get_bge_model()
    
    # 
    test_texts = [
        "",
        "",
        "",
        "AI",
        ""
    ]
    
    print(" :")
    for i, text in enumerate(test_texts, 1):
        print(f"   {i}. {text}")
    
    # 
    print(f"\n  {len(test_texts)} ...")
    embeddings = bge_model.batch_encode(test_texts)
    
    print(f" !")
    print(f"    : {embeddings.shape}")
    print(f"    : {embeddings.shape[1]}")
    print(f"    : {[f'{np.linalg.norm(emb):.4f}' for emb in embeddings[:3]]}")
    
    # 
    print(f"\n :")
    similarities = bge_model.compute_similarity(embeddings[0:1], embeddings)
    
    print(f"   : '{test_texts[0]}'")
    print(f"   :")
    
    for i, (text, sim) in enumerate(zip(test_texts, similarities[0])):
        print(f"      {i+1}. {sim:.4f} - {text}")
    
    # 
    print(f"\n :")
    query = ""
    query_embedding = bge_model.encode_query(query)
    
    print(f"   : {query}")
    print(f"   : {query_embedding.shape}")
    print(f"   : {np.linalg.norm(query_embedding):.4f}")
    
    # 
    doc_similarities = bge_model.compute_similarity(
        query_embedding.reshape(1, -1), 
        embeddings
    )
    
    print(f"   :")
    for i, (text, sim) in enumerate(zip(test_texts, doc_similarities[0])):
        print(f"      {i+1}. {sim:.4f} - {text}")


def interactive_qa_mode(pipeline: RAGFlowPipeline):
    """"""
    
    print("\n" + "="*80)
    print(" RAGFlow + BGE ")
    print("="*80)
    
    print(" !")
    print(" :  'quit'  'exit' ")
    print(" :  'help' ")
    
    while True:
        try:
            user_input = input("\n : ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '']:
                print(" !")
                break
            
            if user_input.lower() in ['help', '']:
                print("""
 :
• RAGFlow
• BGE-large-zh-v1.5
• Milvus
• Qwen-Plus

 :
• ""
• ""
• ""
• "AI"
                """)
                continue
            
            # RAG
            print(f"\n ...")
            result = pipeline.query(user_input, top_k=3)
            
            # 
            print(f"\n :")
            print(f"   {result['answer']}")
            
            print(f"\n :")
            print(f"   ⏱  : {result['total_time']:.2f}")
            print(f"    : {result['stats']['retrieved_chunks']}")
            print(f"    : {result['stats']['max_similarity']:.4f}")
            
        except KeyboardInterrupt:
            print("\n\n ")
            break
        except Exception as e:
            print(f"\n : {e}")


def main():
    """"""
    
    print_banner()
    
    # 
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "full"
    
    try:
        if mode == "process":
            # 
            pipeline = demo_document_processing()
            
        elif mode == "qa":
            # 
            print("  ...")
            pipeline = RAGFlowPipeline()
            demo_intelligent_qa(pipeline)
            
        elif mode == "chunk":
            # 
            demo_chunking_strategies()
            return
            
        elif mode == "embed":
            # 
            demo_bge_embedding()
            return
            
        elif mode == "interactive":
            # 
            pipeline = RAGFlowPipeline()
            interactive_qa_mode(pipeline)
            return
            
        else:
            # 
            pipeline = demo_document_processing()
            demo_intelligent_qa(pipeline)
            demo_chunking_strategies()
            demo_bge_embedding()
            
            # 
            user_choice = input("\n ? (y/n): ").strip().lower()
            if user_choice in ['y', 'yes', '']:
                interactive_qa_mode(pipeline)
    
    except Exception as e:
        print(f"\n : {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n RAGFlow + BGE !")


if __name__ == "__main__":
    main() 