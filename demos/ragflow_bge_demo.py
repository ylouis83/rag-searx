#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow + BGE向量化演示系统
基于RAGFlow文档处理框架和BGE-large-zh-v1.5向量模型的RAG演示
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入RAGFlow + BGE相关模块
from ragflow_integration.ragflow_rag_pipeline import RAGFlowPipeline
from ragflow_integration.document_processor import ChunkConfig, ChunkingStrategy


def print_banner():
    """打印系统横幅"""
    
    banner = """

                     🔧 RAGFlow + BGE演示系统 🚀                           
                                                                              
   🛠 技术架构：                                                                
     • RAGFlow文档处理框架 (智能分块/关键词/问题生成)                          
     • BGE-large-zh-v1.5向量模型 (1024维中文语义向量)                        
     • Milvus向量数据库 + 相似度检索                                     
     • Qwen-Plus大语言模型智能问答                                           
                                                                              
   🎯 核心特性: Python + RAGFlow + BGE + Milvus + DashScope API                

    """
    
    print(banner)


def demo_document_processing():
    """演示RAGFlow文档处理功能"""
    
    print("\n" + "="*80)
    print("📄 RAGFlow + BGE文档处理演示")
    print("="*80)
    
    # 配置RAGFlow分块策略
    chunk_config = ChunkConfig(
        strategy=ChunkingStrategy.BOOK,     # 书籍分块策略
        chunk_token_count=256,              # 每个chunk 256 tokens
        chunk_overlap=0.1,                  # 10% 重叠率
        auto_keywords=True,                 # 自动生成关键词
        auto_question=True                  # 自动生成问题
    )
    
    # 初始化RAGFlow管道
    print("🔧 正在初始化RAGFlow + BGE处理管道...")
    pipeline = RAGFlowPipeline(
        chunk_config=chunk_config,
        embedding_model_name="BAAI/bge-large-zh-v1.5",
        collection_name="ragflow_bge_demo"
    )
    
    # 处理文档 - ragtest.pdf
    target_file = "test_data/ragtest.pdf"
    
    if Path(target_file).exists():
        print(f"\n📁 正在处理文档: {target_file}")
        success = pipeline.process_and_store_document_with_details(target_file)
        
        if success:
            print(f"✅ 文档 {target_file} 处理完成")
        else:
            print(f"❌ 文档 {target_file} 处理失败")
    else:
        print(f"⚠️ 文档不存在: {target_file}")
        print("💡 建议: 请在test_data目录下放置ragtest.pdf文件")
    
    return pipeline


def demo_intelligent_qa(pipeline: RAGFlowPipeline):
    """演示智能问答功能"""
    
    print("\n" + "="*80)
    print("🤖 RAGFlow + BGE智能问答演示")
    print("="*80)
    
    # 测试查询列表
    test_queries = [
        "深度学习的基本概念是什么",
        "神经网络的训练过程",
        "AlexNet网络结构特点",
        "卷积神经网络的优势",
        "机器学习与人工智能的关系",
        "自然语言处理的应用场景"
    ]
    
    print("🚀 开始智能问答测试...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"❓ 测试问题 {i}/{len(test_queries)}")
        print(f"{'='*60}")
        
        # 执行RAG查询
        result = pipeline.query(query, top_k=3)
        
        # 显示检索结果
        print(f"\n🔍 检索结果:")
        for j, search_result in enumerate(result["search_results"], 1):
            print(f"\n     📄 结果 {j}:")
            print(f"       文件名: {search_result['file_name']}")
            print(f"       页码: {search_result['page_number']}")
            print(f"       章节: {search_result['chapter_title']}")
            print(f"        关键词: {', '.join(search_result['keywords'][:3])}")
            print(f"       相似度: {search_result['score']:.4f}")
            print(f"       内容摘要: {search_result['content'][:100]}...")
        
        print(f"\n📊 查询统计:")
        merged_content = "\n\n".join([r['content'] for r in result["search_results"]])
        print(f"    合并内容长度: {len(merged_content)} 个字符")
        print(f"    涉及文件: {set([r['file_name'] for r in result['search_results']])}")
        
        # 等待用户确认
        if i < len(test_queries):
            input(f"\n⏳ 按Enter键继续下一个测试...")


def demo_chunking_strategies():
    """演示RAGFlow分块策略"""
    
    print("\n" + "="*80)
    print("📝 RAGFlow分块策略演示")
    print("="*80)
    
    # 不同分块策略配置
    strategies = [
        (ChunkingStrategy.BOOK, "书籍策略 - 适合长文本和教科书"),
        (ChunkingStrategy.PAPER, "论文策略 - 适合学术论文和研究报告"),
        (ChunkingStrategy.QA, "问答策略 - 适合FAQ和问答文档"),
        (ChunkingStrategy.TABLE, "表格策略 - 适合结构化数据和表格")
    ]
    
    test_text = """
人工智能技术发展概述

第一章 人工智能基础

人工智能（Artificial Intelligence，AI）是计算机科学的一个重要分支，旨在创建能够执行通常需要人类智能的任务的机器和软件。

1.1 人工智能定义

自1956年达特茅斯会议首次提出"人工智能"概念以来，AI技术经历了多次发展浪潮。现代人工智能包括机器学习、深度学习、自然语言处理、计算机视觉等多个领域。

第二章 机器学习基础

机器学习是人工智能的核心技术之一，通过算法让计算机系统从数据中自动学习和改进，无需明确编程。

2.1 监督学习

监督学习使用标记的训练数据来训练模型，包括分类和回归任务。常见算法包括决策树、支持向量机、神经网络等。

    """
    
    from ragflow_integration.document_processor import RAGFlowDocumentProcessor
    
    for strategy, description in strategies:
        print(f"\n🎯 测试策略: {description}")
        print(f"   策略类型: {strategy.value}")
        
        # 创建分块配置
        config = ChunkConfig(
            strategy=strategy,
            chunk_token_count=128,
            chunk_overlap=0.1,
            auto_keywords=True,
            auto_question=True
        )
        
        # 创建文档处理器
        processor = RAGFlowDocumentProcessor(config)
        
        try:
            # 执行分块处理
            chunks = processor._chunk_text_by_strategy(test_text)
            
            print(f"    生成分块数: {len(chunks)}个")
            
            for i, chunk in enumerate(chunks[:2], 1):  # 显示前2个分块
                print(f"\n     📄 分块 {i}:")
                print(f"       分块ID: {chunk.chunk_id[:16]}...")
                print(f"       文本内容: {chunk.content[:150]}...")
                print(f"        关键词: {chunk.keywords}")
                print(f"       生成问题: {chunk.questions}")
                
        except Exception as e:
            print(f"    处理出错: {e}")


def demo_bge_embedding():
    """演示BGE向量化功能"""
    
    print("\n" + "="*80)
    print("🔤 BGE-large-zh-v1.5向量化演示")
    print("="*80)
    
    from ragflow_integration.bge_embedding import get_bge_model
    
    # 获取BGE模型实例
    bge_model = get_bge_model()
    
    # 测试文本列表
    test_texts = [
        "深度学习是机器学习的重要分支",
        "神经网络通过反向传播算法进行训练",
        "卷积神经网络在图像识别中表现优异",
        "自然语言处理涉及文本理解和生成",
        "人工智能技术正在改变我们的生活"
    ]
    
    print("📝 测试文本列表:")
    for i, text in enumerate(test_texts, 1):
        print(f"   {i}. {text}")
    
    # 批量向量化
    print(f"\n🔄 正在对 {len(test_texts)} 条文本进行向量化...")
    embeddings = bge_model.batch_encode(test_texts)
    
    print(f"✅ 向量化完成!")
    print(f"    输出维度: {embeddings.shape}")
    print(f"    向量长度: {embeddings.shape[1]}维")
    print(f"    向量模长: {[f'{np.linalg.norm(emb):.4f}' for emb in embeddings[:3]]}")
    
    # 相似度计算演示
    print(f"\n🔗 文本相似度计算:")
    similarities = bge_model.compute_similarity(embeddings[0:1], embeddings)
    
    print(f"   基准文本: '{test_texts[0]}'")
    print(f"   与其他文本的相似度:")
    
    for i, (text, sim) in enumerate(zip(test_texts, similarities[0])):
        print(f"      {i+1}. {sim:.4f} - {text}")
    
    # 查询向量化演示
    print(f"\n❓ 查询向量化演示:")
    query = "机器学习算法有哪些类型"
    query_embedding = bge_model.encode_query(query)
    
    print(f"   查询文本: {query}")
    print(f"   查询向量维度: {query_embedding.shape}")
    print(f"   查询向量模长: {np.linalg.norm(query_embedding):.4f}")
    
    # 查询与文档的相似度
    doc_similarities = bge_model.compute_similarity(
        query_embedding.reshape(1, -1), 
        embeddings
    )
    
    print(f"   查询与文档的相似度:")
    for i, (text, sim) in enumerate(zip(test_texts, doc_similarities[0])):
        print(f"      {i+1}. {sim:.4f} - {text}")


def interactive_qa_mode(pipeline: RAGFlowPipeline):
    """交互式问答模式"""
    
    print("\n" + "="*80)
    print("🚀 RAGFlow + BGE交互式问答")
    print("="*80)
    
    print("🎯 欢迎使用交互式问答模式!")
    print("💡 提示: 输入 'quit' 或 'exit' 退出")
    print("💡 提示: 输入 'help' 查看使用帮助")
    
    while True:
        try:
            user_input = input("\n❓ 请输入您的问题: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 感谢使用问答系统!")
                break
            
            if user_input.lower() in ['help', '帮助']:
                print("""
🤖 RAGFlow + BGE问答系统功能:
• 基于RAGFlow文档处理框架
• 使用BGE-large-zh-v1.5向量模型
• 支持Milvus向量数据库检索
• 集成Qwen-Plus大语言模型

📝 建议问题类型:
• "深度学习的基本原理是什么"
• "如何选择合适的机器学习算法"
• "神经网络训练中的常见问题"
• "人工智能在哪些领域有应用"
                """)
                continue
            
            # 执行RAG查询
            print(f"\n🔍 正在搜索相关内容...")
            result = pipeline.query(user_input, top_k=3)
            
            # 显示回答
            print(f"\n🤖 智能回答:")
            print(f"   {result['answer']}")
            
            print(f"\n📊 查询统计:")
            print(f"   ⏱ 响应时间: {result['total_time']:.2f}秒")
            print(f"    检索分块数: {result['stats']['retrieved_chunks']}")
            print(f"    最大相似度: {result['stats']['max_similarity']:.4f}")
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出交互模式")
            break
        except Exception as e:
            print(f"\n❌ 处理过程中出错: {e}")


def main():
    """主函数"""
    
    print_banner()
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "full"
    
    try:
        if mode == "process":
            # 仅演示文档处理
            pipeline = demo_document_processing()
            
        elif mode == "qa":
            # 仅演示问答功能
            print("🔧 正在初始化系统...")
            pipeline = RAGFlowPipeline()
            demo_intelligent_qa(pipeline)
            
        elif mode == "chunk":
            # 仅演示分块策略
            demo_chunking_strategies()
            return
            
        elif mode == "embed":
            # 仅演示向量化功能
            demo_bge_embedding()
            return
            
        elif mode == "interactive":
            # 仅运行交互模式
            pipeline = RAGFlowPipeline()
            interactive_qa_mode(pipeline)
            return
            
        else:
            # 完整演示流程
            pipeline = demo_document_processing()
            demo_intelligent_qa(pipeline)
            demo_chunking_strategies()
            demo_bge_embedding()
            
            # 询问是否进入交互模式
            user_choice = input("\n🤔 是否进入交互式问答体验? (y/n): ").strip().lower()
            if user_choice in ['y', 'yes', '是']:
                interactive_qa_mode(pipeline)
    
    except Exception as e:
        print(f"\n❌ 系统运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 RAGFlow + BGE演示完成!")


if __name__ == "__main__":
    main() 