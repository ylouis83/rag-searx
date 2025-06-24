#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频RAG演示系统
YouTube视频处理和RAG检索演示
"""

import sys
import time
from pathlib import Path
import json

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragflow_integration.video_metadata_schema import VideoMetadata, SemanticChunk, VideoLanguage, EXAMPLE_DIABETES_VIDEO, display_video_metadata_detailed
from ragflow_integration.video_rag_pipeline import VideoRAGPipeline


def print_banner():
    """打印系统横幅123"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🎬 视频RAG演示系统 🚀                              ║
║                     YouTube视频处理和智能检索演示                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   🔧 技术栈：                                                                ║
║     • YouTube视频处理                                                        ║
║     • BGE向量化模型                                                          ║
║     • Milvus向量数据库                                                       ║
║     • Qwen大语言模型                                                         ║
║                                                                              ║
║   🚀 核心功能：RAGFlow + BGE + Milvus + Qwen + 多路召回                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def demo_video_metadata_schema():
    """演示视频元数据Schema"""
    print("\n" + "="*80)
    print("📊 视频元数据Schema演示")
    print("="*80)
    
    print("📹 视频基本信息:")
    print("-" * 60)
    print(f"视频ID: {EXAMPLE_DIABETES_VIDEO.video_id}")
    print(f"视频URL: {EXAMPLE_DIABETES_VIDEO.video_url}")
    print(f"频道名称: {EXAMPLE_DIABETES_VIDEO.channel_name}")
    print(f"视频分类: {EXAMPLE_DIABETES_VIDEO.category}")
    print(f"⏱ 视频时长: {EXAMPLE_DIABETES_VIDEO.duration_seconds}秒")
    print(f"视频语言: {EXAMPLE_DIABETES_VIDEO.language.value}")
    
    print(f"\n🤖 AI生成摘要:")
    print(f"   {EXAMPLE_DIABETES_VIDEO.ai_summary}")
    
    print(f"\n🏷 关键词标签:")
    print(f"   {', '.join(EXAMPLE_DIABETES_VIDEO.keywords)}")
    
    print(f"\n📝 语义分块 ({len(EXAMPLE_DIABETES_VIDEO.semantic_chunks)}个):")
    for i, chunk in enumerate(EXAMPLE_DIABETES_VIDEO.semantic_chunks, 1):
        print(f"   {i}. {chunk.time_format()} - {chunk.text[:80]}...")
    
    print(f"\n🔍 可向量化内容:")
    vectorizable_content = EXAMPLE_DIABETES_VIDEO.get_vectorizable_content()
    for i, content in enumerate(vectorizable_content, 1):
        print(f"   {i}. {content[:100]}...")


def demo_video_storage(pipeline: VideoRAGPipeline):
    """演示视频数据存储"""
    print("\n" + "="*80)
    print("💾 视频数据存储演示")
    print("="*80)
    
    print("正在存储视频元数据...")
    success = pipeline.store_video_metadata(EXAMPLE_DIABETES_VIDEO)
    
    if success:
        print("✅ 视频数据存储成功!")
        print("\n📊 存储统计:")
        print(f"    存储视频数量: 1个")
        print(f"    语义分块数量: {len(EXAMPLE_DIABETES_VIDEO.semantic_chunks)}个")
        print(f"    向量条目总数: {len(EXAMPLE_DIABETES_VIDEO.semantic_chunks) + 2}个 (分块+摘要+关键词)")
    else:
        print("❌ 视频数据存储失败!")
        return False
    
    return True


def demo_video_search(pipeline: VideoRAGPipeline):
    """演示视频搜索功能"""
    print("\n" + "="*80)
    print("🔍 视频搜索演示")
    print("="*80)
    
    test_queries = [
        "糖尿病治疗方法",
        "血糖控制",
        "饮食建议",
        "运动疗法"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 搜索测试 {i}: {query}")
        print("-" * 50)
        
        search_results = pipeline.search_videos(query, top_k=3)
        
        if search_results:
            print(f"找到 {len(search_results)} 个相关结果:")
            
            for j, result in enumerate(search_results, 1):
                print(f"\n     📺 结果 {j}:")
                print(f"       标题: {result['title']}")
                print(f"       频道: {result['channel_name']}")
                print(f"       类型: {result['content_type']}")
                print(f"       相似度: {result['score']:.4f}")
                
                if result['content_type'] == 'chunk':
                    start_min = result['chunk_start_time'] // 60
                    start_sec = result['chunk_start_time'] % 60
                    end_min = result['chunk_end_time'] // 60
                    end_sec = result['chunk_end_time'] % 60
                    print(f"      ⏰ 时间段: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}")
                
                print(f"       内容摘要: {result['chunk_text'][:100]}...")
        else:
            print("❌ 未找到相关结果")
        
        if i < len(test_queries):
            input("\n⏳ 按Enter继续下一个搜索测试...")


def demo_intelligent_qa(pipeline: VideoRAGPipeline):
    """演示智能问答系统"""
    print("\n" + "="*80)
    print("🤖 智能问答演示")
    print("="*80)
    
    print("🎯 现在开始智能问答演示!")
    print("💡 提示: 输入 'quit' 退出，输入 'help' 查看帮助")
    print("📝 建议问题示例请输入 'help' 查看")
    
    question_count = 0
    
    while True:
        try:
            user_input = input(f"\n❓ 请输入您的问题 ({question_count + 1}): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 感谢使用智能问答系统!")
                break
            
            if user_input.lower() in ['help', '帮助', 'h']:
                print("""
🤖 智能问答系统帮助:
• 基于YouTube视频内容的智能问答
• 支持BGE语义检索
• 使用Milvus向量数据库
• 基于Qwen大语言模型生成答案

📝 建议问题示例:
• "糖尿病患者应该如何控制血糖?"
• "糖尿病的饮食建议有哪些?"
• "什么运动适合糖尿病患者?"
• "糖尿病的早期症状是什么?"
                """)
                continue
            
            question_count += 1
            print(f"\n🔄 正在处理第 {question_count} 个问题...")
            print("="*60)
            
            # 执行RAG查询
            result = pipeline.query(user_input, top_k=3)
            
            print(f"\n🤖 AI智能回答:")
            print("-" * 40)
            print(result["answer"])
            print("-" * 40)
            
            print(f"\n📊 查询统计信息:")
            print(f"   ⏱ 总耗时: {result['total_time']:.2f}秒")
            print(f"   📺 检索到的视频: {result['stats']['retrieved_videos']}个")
            print(f"   📝 检索到的分块: {result['stats']['retrieved_chunks']}个")
            print(f"   🎯 最大相似度: {result['stats']['max_similarity']:.4f}")
            
            # 显示检索详情
            if result.get('search_results') and len(result['search_results']) > 0:
                print(f"\n" + "="*80)
                print("🔍 检索结果详情")
                print("="*80)
                
                # 显示视频信息
                print("📺 相关视频信息:")
                display_video_metadata_detailed(EXAMPLE_DIABETES_VIDEO, show_full_content=False)
                
                print(f"\n📝 相关内容片段:")
                for i, search_result in enumerate(result['search_results'][:3], 1):  # 显示前3个结果
                    print(f"\n     📄 片段 {i}:")
                    print(f"       内容类型: {search_result.get('content_type', 'unknown')}")
                    print(f"       相似度分数: {search_result.get('score', 0):.4f}")
                    
                    if search_result.get('content_type') == 'chunk':
                        start_time = search_result.get('chunk_start_time', 0)
                        end_time = search_result.get('chunk_end_time', 0)
                        start_min = start_time // 60
                        start_sec = start_time % 60
                        end_min = end_time // 60
                        end_sec = end_time % 60
                        print(f"      ⏰ 时间段: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}")
                        print(f"       文本内容: {search_result.get('chunk_text', '')[:120]}...")
                    elif search_result.get('content_type') == 'summary':
                        print(f"       摘要内容: {search_result.get('chunk_text', '')[:120]}...")
                    elif search_result.get('content_type') == 'keywords':
                        print(f"        关键词: {search_result.get('chunk_text', '')}")
            
            print(f"\n✅ 已完成第 {question_count} 个问题的回答")
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出问答系统")
            break
        except Exception as e:
            print(f"\n❌ 处理问题时出错: {e}")
            continue


def demo_prompt_engineering():
    """演示提示词工程"""
    print("\n" + "="*80)
    print("🛠 提示词工程演示 (Qwen模型)")
    print("="*80)
    
    print("📝 提示词结构说明:")
    print("-" * 60)
    
    prompt_structure = """
1. 角色定义 (Role):
   - 专业的AI视频内容助手
   - 基于视频内容提供准确回答

2. 上下文信息 (Context):
   - 来自YouTube视频的结构化内容
   - AI生成的摘要和语义分块

3. 指令要求 (Instructions):
   - 基于提供的视频内容回答问题
   - 保持回答的准确性和相关性
   - 如果内容不足，诚实说明
   - 提供具体的时间戳引用

4. 查询问题 (Query):
   - 用户的具体问题
"""
    
    print(prompt_structure)
    
    print("\n🎯 提示词优化要点:")
    print("     ✅ 角色定义明确 - 专业AI助手")
    print("     ✅ 上下文结构化 - AI摘要 + 语义分块")  
    print("     ✅ 指令清晰具体 - 4个核心要求")
    print("     ✅ 响应格式规范 - 结构化输出")
    print("     ✅ 错误处理机制 - 内容不足时的处理")
    print("     ✅ 引用追踪能力 - 提供时间戳定位")


def interactive_qa_mode(pipeline: VideoRAGPipeline):
    """交互式问答模式"""
    print("\n" + "="*80)
    print("🚀 交互式RAG问答体验")
    print("="*80)
    
    print("🎯 欢迎使用交互式问答模式!")
    print("💡 提示: 输入 'quit' 退出，输入 'help' 查看帮助")
    
    while True:
        try:
            user_input = input("\n❓ 请输入问题: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 感谢使用交互式问答系统!")
                break
            
            if user_input.lower() in ['help', '帮助']:
                print("""
🤖 交互式问答系统:
• 基于YouTube视频内容的实时问答
• 支持BGE语义检索技术
• 使用Milvus向量数据库
• 基于Qwen大语言模型生成回答

📝 问题示例:
• "糖尿病的主要症状有哪些?"
• "如何预防糖尿病?"
• "糖尿病患者的饮食注意事项"
• "适合糖尿病患者的运动方式"
                """)
                continue
            
            # 执行RAG查询
            print(f"\n🔄 正在思考您的问题...")
            result = pipeline.query(user_input, top_k=3)
            
            # 显示回答结果
            print(f"\n🤖 智能回答:")
            print("═" * 60)
            print(result['answer'])
            print("═" * 60)
            
            print(f"\n📊 本次查询统计:")
            print(f"   ⏱ 响应时间: {result['total_time']:.2f}秒")
            print(f"   📝 相关片段: {result['stats']['retrieved_chunks']}个")
            print(f"   🎯 最高相似度: {result['stats']['max_similarity']:.4f}")
            
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
        if mode == "schema":
            # 仅演示Schema
            demo_video_metadata_schema()
            
        elif mode == "prompt":
            # 仅演示提示词工程
            demo_prompt_engineering()
            
        elif mode == "interactive":
            # 仅运行交互模式
            print("🚀 正在初始化RAG系统...")
            pipeline = VideoRAGPipeline()
            interactive_qa_mode(pipeline)
            
        elif mode == "qa":
            # 仅运行问答演示
            print("🚀 正在初始化RAG系统...")
            pipeline = VideoRAGPipeline()
            demo_intelligent_qa(pipeline)
            
        else:
            # 完整演示流程
            print("🚀 开始完整RAG演示流程...")
            
            # 1. Schema演示
            demo_video_metadata_schema()
            input("\n⏳ 按Enter继续下一步...")
            
            # 2. 初始化并存储数据
            pipeline = VideoRAGPipeline()
            storage_success = demo_video_storage(pipeline)
            
            if not storage_success:
                print("❌ 数据存储失败，演示终止")
                return
            
            input("\n⏳ 按Enter继续下一步...")
            
            # 3. 搜索演示
            demo_video_search(pipeline)
            input("\n⏳ 按Enter继续下一步...")
            
            # 4. 智能问答演示
            demo_intelligent_qa(pipeline)
            input("\n⏳ 按Enter继续下一步...")
            
            # 5. 提示词工程演示
            demo_prompt_engineering()
            
            # 6. 交互模式选择
            user_choice = input("\n🤔 是否进入交互式问答体验? (y/n): ").strip().lower()
            if user_choice in ['y', 'yes', '是']:
                interactive_qa_mode(pipeline)
    
    except Exception as e:
        print(f"\n❌ 系统运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 视频RAG演示系统运行完毕!")


if __name__ == "__main__":
    main() 
