#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多路召回RAG演示系统
基于多种召回策略的智能检索演示
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragflow_integration.video_metadata_schema import VideoMetadata, SemanticChunk, VideoLanguage, EXAMPLE_DIABETES_VIDEO, display_video_metadata_detailed
from ragflow_integration.enhanced_video_rag_pipeline import EnhancedVideoRAGPipeline
from ragflow_integration.multi_path_recall import MultiRecallConfig


def print_banner():
    """打印系统横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🔍 多路召回RAG演示系统 🚀                            ║
║                     基于多种召回策略的智能检索演示                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   🛠 召回策略：                                                               ║
║     • 向量召回 (BM25算法)                                                    ║
║     • 关键词召回 (BGE向量模型)                                               ║
║     • 语义召回 (语义理解)                                                    ║
║     • 混合召回 (多策略融合)                                                  ║
║                                                                              ║
║   🚀 核心特性：多路召回 + 策略对比 + 智能融合 + 性能分析                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def demo_multi_recall_comparison():
    """演示多路召回对比"""
    print("\n" + "="*80)
    print("📊 单路召回 vs 多路召回对比演示")
    print("="*80)
    
    # 配置不同的召回策略
    
    # 1. 单路召回配置（仅向量）
    single_config = MultiRecallConfig(
        vector_weight=1.0,
        keyword_weight=0.0,
        semantic_weight=0.0,
        vector_top_k=10,
        keyword_top_k=0,
        semantic_top_k=0,
        final_top_k=5,
        enable_query_expansion=False
    )
    
    # 2. 多路召回配置（混合策略）
    multi_config = MultiRecallConfig(
        vector_weight=0.4,
        keyword_weight=0.4,
        semantic_weight=0.2,
        vector_top_k=15,
        keyword_top_k=10,
        semantic_top_k=8,
        final_top_k=5,
        enable_query_expansion=True,
        enable_synonym_expansion=True
    )
    
    # 初始化管道
    print("🔧 正在初始化单路召回管道...")
    single_pipeline = EnhancedVideoRAGPipeline(recall_config=single_config)
    
    print("🔧 正在初始化多路召回管道...")
    multi_pipeline = EnhancedVideoRAGPipeline(recall_config=multi_config)
    
    # 存储测试数据
    print("\n💾 正在存储测试数据...")
    single_pipeline.store_video_metadata(EXAMPLE_DIABETES_VIDEO)
    multi_pipeline.store_video_metadata(EXAMPLE_DIABETES_VIDEO)
    
    # 测试查询
    test_queries = [
        "糖尿病的症状表现",
        "如何控制血糖水平",
        "糖尿病患者饮食建议",
        "适合糖尿病患者的运动"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n" + "="*60)
        print(f"🔍 测试查询 {i}: {query}")
        print("="*60)
        
        # 单路召回测试
        print("\n📍 单路召回结果:")
        print("-" * 30)
        single_result = single_pipeline.enhanced_query(query, top_k=5)
        single_paths = single_result.get("recall_paths", [])
        single_diversity = single_result.get("stats", {}).get("recall_diversity", 0)
        
        print(f"   召回路径: {', '.join(single_paths)}")
        print(f"   召回多样性: {single_diversity}")
        print(f"   检索文档数: {single_result.get('stats', {}).get('retrieved_documents', 0)}个")
        print(f"   最大相似度: {single_result.get('stats', {}).get('max_similarity', 0):.4f}")
        
        # 多路召回测试
        print("\n🎯 多路召回结果:")
        print("-" * 30)
        multi_result = multi_pipeline.enhanced_query(query, top_k=5)
        multi_paths = multi_result.get("recall_paths", [])
        multi_diversity = multi_result.get("stats", {}).get("recall_diversity", 0)
        
        print(f"   召回路径: {', '.join(multi_paths)}")
        print(f"   召回多样性: {multi_diversity}")
        print(f"   检索文档数: {multi_result.get('stats', {}).get('retrieved_documents', 0)}个")
        print(f"   最大相似度: {multi_result.get('stats', {}).get('max_similarity', 0):.4f}")
        
        # 对比分析
        print(f"\n📈 性能提升分析:")
        diversity_improvement = multi_diversity - single_diversity
        doc_improvement = multi_result.get('stats', {}).get('retrieved_documents', 0) - single_result.get('stats', {}).get('retrieved_documents', 0)
        
        print(f"   多样性提升: {diversity_improvement:.2f} 分")
        print(f"   文档数提升: {doc_improvement} 个")
        
        if i < len(test_queries):
            input("\n⏳ 按Enter继续下一个测试...")


def demo_recall_path_analysis():
    """演示召回路径分析"""
    print("\n" + "="*80)
    print("🔬 召回路径详细分析")
    print("="*80)
    
    # 配置多路召回
    config = MultiRecallConfig(
        vector_weight=0.3,
        keyword_weight=0.4,
        semantic_weight=0.3,
        vector_top_k=10,
        keyword_top_k=8,
        semantic_top_k=6,
        final_top_k=8,
        enable_query_expansion=True,
        enable_synonym_expansion=True
    )
    
    pipeline = EnhancedVideoRAGPipeline(recall_config=config)
    pipeline.store_video_metadata(EXAMPLE_DIABETES_VIDEO)
    
    # 分析查询
    analysis_query = "糖尿病患者的血糖管理方法"
    print(f"🔍 分析查询: {analysis_query}")
    
    result = pipeline.enhanced_query(analysis_query, top_k=8)
    
    # 路径统计分析
    print(f"\n📊 各召回路径统计:")
    path_stats = result.get("path_statistics", {})
    
    for path, stats in path_stats.items():
        print(f"\n🎯 {path.upper()}路径:")
        print(f"    检索数量: {stats['count']}个")
        print(f"    平均分数: {stats['avg_score']:.4f}")
        print(f"    最高分数: {stats['max_score']:.4f}")
        
        # 显示该路径的前3个结果
        path_results = [r for r in result['search_results'] if r.get('recall_path') == path]
        for i, res in enumerate(path_results[:3], 1):  # 显示前3个结果
            content_preview = res.get('content', '')[:80] + "..." if len(res.get('content', '')) > 80 else res.get('content', '')
            print(f"     {i}. 分数: {res.get('score', 0):.4f} | 类型: {res.get('content_type', '')} | 内容: {content_preview}")
    
    # 整体统计
    print(f"\n📈 整体召回统计:")
    print(f"    总检索数量: {len(result['search_results'])}个")
    print(f"    召回多样性: {result['stats']['recall_diversity']:.2f}分")
    print(f"    平均相似度: {result['stats']['avg_similarity']:.4f}")
    print(f"    最大相似度: {result['stats']['max_similarity']:.4f}")


def demo_interactive_multi_recall():
    """演示交互式多路召回"""
    print("\n" + "="*80)
    print("🚀 交互式多路召回体验")
    print("="*80)
    
    # 配置多路召回
    config = MultiRecallConfig(
        vector_weight=0.35,
        keyword_weight=0.35,
        semantic_weight=0.3,
        vector_top_k=12,
        keyword_top_k=10,
        semantic_top_k=8,
        final_top_k=10,
        enable_query_expansion=True,
        enable_synonym_expansion=True
    )
    
    print("🔧 正在初始化多路召回系统...")
    pipeline = EnhancedVideoRAGPipeline(recall_config=config)
    pipeline.store_video_metadata(EXAMPLE_DIABETES_VIDEO)
    
    print("🎯 欢迎使用交互式多路召回系统!")
    print("💡 提示: 输入 'quit' 退出，输入 'help' 查看帮助，输入 'stats' 查看系统统计")
    
    question_count = 0
    
    while True:
        try:
            user_input = input(f"\n❓ 请输入查询问题 ({question_count + 1}): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 感谢使用多路召回系统!")
                break
            
            if user_input.lower() in ['help', '帮助', 'h']:
                print("""
🤖 多路召回系统帮助:
• 向量召回: 基于BM25算法的精确匹配
• 关键词召回: 基于BGE向量模型的语义检索  
• 语义召回: 基于深度语义理解的智能匹配
• 混合召回: 多策略融合的综合检索

📝 建议查询示例:
• "糖尿病的早期症状有哪些"
• "如何有效控制血糖水平"
• "糖尿病患者的饮食禁忌"
• "适合糖尿病患者的运动方式"
                """)
                continue
            
            if user_input.lower() in ['stats', '统计', 's']:
                stats = pipeline.get_system_stats()
                print("\n📊 系统统计信息:")
                print(f"    存储视频数: {stats['videos_stored']} 个")
                print(f"    索引文档数: {stats['documents_indexed']} 个")
                print(f"    嵌入模型: {stats['embedding_model']}")
                print(f"     权重配置: 向量{config.vector_weight} | 关键词{config.keyword_weight} | 语义{config.semantic_weight}")
                continue
            
            question_count += 1
            print(f"\n🔄 正在处理查询 ({question_count})...")
            print("="*60)
            
            # 执行多路召回查询
            result = pipeline.enhanced_query(user_input, top_k=8)
            
            print(f"\n🤖 智能回答:")
            print("-" * 50)
            print(result["answer"])
            print("-" * 50)
            
            # 召回统计信息
            print(f"\n📊 召回统计信息:")
            print(f"   ⏱ 总耗时: {result['total_time']:.2f}秒")
            print(f"   📺 检索视频: {result['stats']['retrieved_videos']}个")
            print(f"   📝 检索文档: {result['stats']['retrieved_documents']}个")
            print(f"   🎯 召回路径: {', '.join(result['recall_paths'])}")
            print(f"   🌟 召回多样性: {result['stats']['recall_diversity']:.2f} 分")
            print(f"   📈 平均相似度: {result['stats']['avg_similarity']:.4f}")
            print(f"   🔥 最大相似度: {result['stats']['max_similarity']:.4f}")
            
            # 路径统计详情
            if result.get('path_statistics'):
                print(f"\n🔬 各路径统计:")
                for path, stats in result['path_statistics'].items():
                    print(f"    {path}路径: {stats['count']}个, 平均{stats['avg_score']:.4f}, 最高{stats['max_score']:.4f}")
            
            # 检索结果详情
            if result.get('search_results') and len(result['search_results']) > 0:
                print(f"\n" + "="*80)
                print("🔍 检索结果详情")
                print("="*80)
                
                print("📺 相关视频信息:")
                display_video_metadata_detailed(EXAMPLE_DIABETES_VIDEO, show_full_content=False)
                
                print(f"\n📝 相关内容片段:")
                for i, search_result in enumerate(result['search_results'][:5], 1):  # 显示前5个结果
                    print(f"\n     📄 片段 {i}:")
                    print(f"       召回路径: {search_result.get('recall_path', 'unknown')}")
                    print(f"       相似度分数: {search_result.get('score', 0):.4f}")
                    print(f"        内容类型: {search_result.get('content_type', 'unknown')}")
                    
                    if search_result.get('content_type') == 'chunk':
                        start_time = search_result.get('chunk_start_time', 0)
                        end_time = search_result.get('chunk_end_time', 0)
                        start_min = start_time // 60
                        start_sec = start_time % 60
                        end_min = end_time // 60
                        end_sec = end_time % 60
                        print(f"      ⏰ 时间段: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}")
                    
                    content_preview = search_result.get('content', '')[:100] + "..." if len(search_result.get('content', '')) > 100 else search_result.get('content', '')
                    print(f"       文本内容: {content_preview}")
            
            print(f"\n✅ 已完成第 {question_count} 个查询")
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出多路召回系统")
            break
        except Exception as e:
            print(f"\n❌ 处理查询时出错: {e}")
            continue


def demo_advanced_configuration():
    """演示高级配置对比"""
    print("\n" + "="*80)
    print("⚙️ 高级召回配置对比演示")
    print("="*80)
    
    print("🔧 正在测试不同配置策略...")
    
    # 不同的配置策略
    configs = {
        "均衡策略": MultiRecallConfig(
            vector_weight=0.33, keyword_weight=0.33, semantic_weight=0.34,
            vector_top_k=10, keyword_top_k=10, semantic_top_k=10, final_top_k=6
        ),
        "向量优先": MultiRecallConfig(
            vector_weight=0.6, keyword_weight=0.25, semantic_weight=0.15,
            vector_top_k=15, keyword_top_k=8, semantic_top_k=5, final_top_k=6
        ),
        "关键词优先": MultiRecallConfig(
            vector_weight=0.2, keyword_weight=0.6, semantic_weight=0.2,
            vector_top_k=8, keyword_top_k=15, semantic_top_k=5, final_top_k=6
        ),
        "语义优先": MultiRecallConfig(
            vector_weight=0.3, keyword_weight=0.2, semantic_weight=0.5,
            vector_top_k=8, keyword_top_k=8, semantic_top_k=15, final_top_k=6
        )
    }
    
    test_query = "糖尿病患者如何控制血糖"
    print(f"🔍 测试查询: {test_query}")
    
    for config_name, config in configs.items():
        print(f"\n🎯 {config_name}配置:")
        print(f"   权重比例: 向量{config.vector_weight} | 关键词{config.keyword_weight} | 语义{config.semantic_weight}")
        
        pipeline = EnhancedVideoRAGPipeline(recall_config=config)
        pipeline.store_video_metadata(EXAMPLE_DIABETES_VIDEO)
        
        result = pipeline.enhanced_search(test_query, top_k=6)
        
        recall_paths = list(set([r.get("recall_path", "") for r in result]))
        avg_score = sum([r.get("score", 0) for r in result]) / len(result) if result else 0
        max_score = max([r.get("score", 0) for r in result]) if result else 0
        
        print(f"    检索结果数: {len(result)} 个")
        print(f"     激活路径: {', '.join(recall_paths)}")
        print(f"    平均分数: {avg_score:.4f}")
        print(f"    最高分数: {max_score:.4f}")


def main():
    """主函数"""
    print_banner()
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "full"
    
    try:
        if mode == "comparison":
            # 仅运行对比演示
            demo_multi_recall_comparison()
            
        elif mode == "analysis":
            # 仅运行路径分析
            demo_recall_path_analysis()
            
        elif mode == "interactive":
            # 仅运行交互模式
            demo_interactive_multi_recall()
            
        elif mode == "config":
            # 仅运行配置演示
            demo_advanced_configuration()
            
        else:
            # 完整演示流程
            print("🚀 开始完整多路召回演示...")
            
            # 1. 对比演示
            demo_multi_recall_comparison()
            input("\n⏳ 按Enter继续下一步...")
            
            # 2. 路径分析
            demo_recall_path_analysis()
            input("\n⏳ 按Enter继续下一步...")
            
            # 3. 配置对比
            demo_advanced_configuration()
            input("\n⏳ 按Enter继续下一步...")
            
            # 4. 交互体验
            demo_interactive_multi_recall()
    
    except Exception as e:
        print(f"\n❌ 系统运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 多路召回演示系统运行完毕!")


if __name__ == "__main__":
    main() 