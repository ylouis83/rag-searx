#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单视频RAG演示 - 基于YouTube视频的智能检索和问答
集成BGE向量模型、Qwen大语言模型和Milvus向量数据库
"""

import sys
import time
import os
import dashscope
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入相关模块
from ragflow_integration.video_metadata_schema import VideoMetadata, SemanticChunk, VideoLanguage, EXAMPLE_DIABETES_VIDEO, display_video_metadata_detailed

# 配置通义千问API
os.environ['DASHSCOPE_API_KEY'] = 'sk-b70842d25c884aa9aa18955b00c24d37'
dashscope.api_key = 'sk-b70842d25c884aa9aa18955b00c24d37'


def print_banner():
    """打印系统横幅"""
    banner = """

                    🎬 简单视频RAG演示系统 🚀                                 
                                                                              
   🔧 技术栈：                                                                
     • YouTube视频处理和元数据提取                                                    
     • BGE向量化模型进行语义检索                                                    
     • Qwen大语言模型生成智能回答                                           
     • Milvus向量数据库存储和检索                                                  
                                                                              
   🎯 核心功能：视频内容理解 + 语义检索 + 智能问答                               

"""
    print(banner)


def get_mock_video_data():
    """获取模拟视频数据"""
    return {
        "video_url": "https://www.youtube.com/watch?v=au-w0QXB6jg",
        "title": "了解2型糖尿病 (Understanding Type 2 Diabetes Mellitus)",
        "channel_name": "Nucleus Medical Media",
        "category": "健康医疗",
        "ai_summary": "这个视频详细介绍了2型糖尿病的基本概念、发病机制、主要症状以及预防和管理方法。内容涵盖了胰岛素抵抗、血糖控制、饮食管理、运动疗法等关键知识点，为糖尿病患者和高危人群提供了实用的健康指导。",
        "keywords": "2型糖尿病, 血糖控制, 胰岛素抵抗, 饮食管理, 运动疗法, 健康生活",
        "semantic_chunks": [
            {
                "start_time_seconds": 46,
                "end_time_seconds": 75,
                "text": "2型糖尿病的主要症状包括频繁口渴、多尿、疲劳、视力模糊、伤口愈合缓慢等。如果不及时诊断和治疗，可能会导致严重的并发症，如心血管疾病、肾病、神经病变和眼部疾病。",
                "similarity_score": 0.8924
            },
            {
                "start_time_seconds": 76,
                "end_time_seconds": 100,
                "text": "饮食管理是2型糖尿病治疗的关键组成部分。患者应该选择低糖、高纤维的食物，控制碳水化合物的摄入量，保持规律的进餐时间。建议多吃蔬菜、瘦肉、全谷物，避免加工食品和含糖饮料。",
                "similarity_score": 0.8456
            },
            {
                "start_time_seconds": 101,
                "end_time_seconds": 130,
                "text": "规律的体育锻炼对2型糖尿病患者非常重要。适量的有氧运动和力量训练可以帮助改善胰岛素敏感性，降低血糖水平，控制体重。建议每周至少进行150分钟的中等强度运动，如快走、游泳或骑自行车。",
                "similarity_score": 0.7892
            }
        ]
    }


def build_qwen_prompt(query: str, video_context: dict) -> str:
    """构建Qwen模型的提示词"""
    
    # 构建语义分块文本
    semantic_chunks_text = ""
    for i, chunk in enumerate(video_context.get("semantic_chunks", []), 1):
        start_min = chunk["start_time_seconds"] // 60
        start_sec = chunk["start_time_seconds"] % 60
        end_min = chunk["end_time_seconds"] // 60
        end_sec = chunk["end_time_seconds"] % 60
        
        semantic_chunks_text += f'''
    {{
      "时间段": "{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}",
      "相似度": {chunk["similarity_score"]:.4f},
      "内容": "{chunk["text"]}"
    }}'''
        if i < len(video_context["semantic_chunks"]):
            semantic_chunks_text += ","
    
    prompt = f"""角色定义 (Role):
你是一个专业的AI视频内容助手，基于YouTube视频内容为用户提供准确、有用的回答。

上下文信息 (Context):
基于视频 "{video_context.get('title', '')}" 的内容信息

[视频元数据]
{{
  "video_url": "{video_context.get('video_url', '')}",
  "title": "{video_context.get('title', '')}",
  "channel_name": "{video_context.get('channel_name', '')}",
  "category": "{video_context.get('category', '')}",
  "ai_summary": "{video_context.get('ai_summary', '')}",
  "keywords": "{video_context.get('keywords', '')}",
  "semantic_chunks": [{semantic_chunks_text}
  ]
}}

任务指令 (Task & Instructions):

请基于以上视频内容信息，为用户提供准确、有用的回答：

1. 优先使用语义分块内容：使用相似度最高的语义分块(semantic_chunks)来回答问题

2. 结合摘要信息：如果需要，可以结合ai_summary和keywords提供更全面的答案

3. 提供时间戳引用：在回答中自然地提及相关的时间段，格式如 "在视频的xx:xx-xx:xx时间段中提到..."

4. 保持诚实回答：如果提供的视频内容不足以完全回答问题，请诚实说明并建议观看完整视频

用户问题 (User's Query):
"{query}"

请提供详细、准确的回答："""

    return prompt


def generate_video_answer(query: str, video_context: dict) -> str:
    """生成基于视频内容的智能回答"""
    try:
        print("🤖 正在生成智能回答...")
        
        # 构建提示词
        prompt = build_qwen_prompt(query, video_context)
        
        print(f"📝 提示词长度: {len(prompt)} 个字符")
        
        # 调用通义千问API
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
            print(f"✅ 回答生成完成: {generate_time:.2f}秒")
            return answer
        else:
            print(f"❌ 调用通义千问失败: {response}")
            return "抱歉，AI服务暂时不可用，请稍后再试。"
            
    except Exception as e:
        print(f"❌ 生成回答时出错: {e}")
        return f"生成回答时遇到错误: {str(e)}"


def demo_video_metadata():
    """演示视频元数据结构"""
    print("\n" + "="*80)
    print("📊 视频元数据结构演示")
    print("="*80)
    
    video_data = get_mock_video_data()
    
    print("📹 基本信息:")
    print("-" * 60)
    print(f"视频标题: {video_data['title']}")
    print(f"视频链接: {video_data['video_url']}")
    print(f"频道名称: {video_data['channel_name']}")
    print(f"视频分类: {video_data['category']}")
    
    print(f"\n🤖 AI生成摘要:")
    print(f"   {video_data['ai_summary']}")
    
    print(f"\n🏷 关键词标签:")
    print(f"   {video_data['keywords']}")
    
    print(f"\n📝 语义分块 ({len(video_data['semantic_chunks'])}个):")
    for i, chunk in enumerate(video_data['semantic_chunks'], 1):
        start_min = chunk["start_time_seconds"] // 60
        start_sec = chunk["start_time_seconds"] % 60
        end_min = chunk["end_time_seconds"] // 60
        end_sec = chunk["end_time_seconds"] % 60
        time_range = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
        
        print(f"   {i}. 时间段: {time_range} | 相似度: {chunk['similarity_score']:.4f}")
        print(f"      内容: {chunk['text']}")
        print()


def demo_prompt_engineering():
    """演示提示词工程设计"""
    print("\n" + "="*80)
    print("🛠 提示词工程设计 (RAGFlow + Qwen)")
    print("="*80)
    
    print("📝 提示词结构设计:")
    print("-" * 60)
    
    structure = """
 1. 角色定义 (Role):
   • 专业的AI视频内容助手
   • 基于视频内容提供准确回答
   • 保持专业性和实用性

 2. 上下文信息 (Context):
   • 结构化的视频元数据
   • AI生成的摘要和关键词
   • 带时间戳的语义分块

 3. 任务指令 (Instructions):
   • 优先使用语义分块内容回答
   • 结合摘要信息提供全面答案
   • 提供时间戳引用便于定位
   • 保持回答的诚实性和准确性

 4. 用户查询 (Query):
   • 用户的具体问题
   • 需要基于视频内容回答
"""
    print(structure)
    
    print("🎯 设计优势:")
    print("     ✅ 角色定位明确 - 专业视频内容助手")
    print("     ✅ 上下文结构化 - JSON格式的视频元数据")  
    print("     ✅ 指令清晰具体 - 4个核心指导原则")
    print("     ✅ 时间戳引用 - 方便用户定位具体内容")
    print("     ✅ 错误处理机制 - 内容不足时的诚实回应")
    print("     ✅ 多维度信息融合 - 摘要+关键词+分块")


def demo_intelligent_qa():
    """演示智能问答系统"""
    print("\n" + "="*80)
    print("🤖 智能问答系统演示")
    print("="*80)
    
    video_data = get_mock_video_data()
    
    print("🎯 欢迎使用智能问答演示!")
    print("💡 提示: 输入 'quit' 退出，输入 'help' 查看帮助")
    print("📝 可以询问关于糖尿病相关的问题")
    
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
🤖 智能问答系统功能:
• 基于YouTube视频内容的智能问答
• 支持BGE语义检索技术
• 使用RAGFlow文档处理框架
• 基于Qwen大语言模型生成回答

📝 建议问题示例:
• "糖尿病的主要症状有哪些?"
• "如何有效控制血糖水平?"
• "糖尿病患者的饮食建议"
• "适合糖尿病患者的运动方式"
                """)
                continue
            
            question_count += 1
            print(f"\n🔄 正在处理第 {question_count} 个问题...")
            print("="*60)
            
            # 搜索相关内容
            print("🔍 正在搜索相关内容...")
            print(f"    视频标题: {video_data['title']}")
            print(f"    最高相似度: {max([chunk['similarity_score'] for chunk in video_data['semantic_chunks']]):.4f}")
            print(f"    分块数量: {len(video_data['semantic_chunks'])}")
            
            # 生成回答
            answer = generate_video_answer(user_input, video_data)
            
            print(f"\n🤖 AI智能回答:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
            # 显示详细信息
            print(f"\n" + "="*80)
            print("📊 详细信息展示")
            print("="*80)
            
            # 显示视频信息
            print("📺 视频信息:")
            display_video_metadata_detailed(EXAMPLE_DIABETES_VIDEO, show_full_content=False)
            
            print(f"\n📝 相关内容分块:")
            for i, chunk in enumerate(video_data['semantic_chunks'], 1):
                start_min = chunk["start_time_seconds"] // 60
                start_sec = chunk["start_time_seconds"] % 60
                end_min = chunk["end_time_seconds"] // 60
                end_sec = chunk["end_time_seconds"] % 60
                time_range = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
                
                print(f"\n     📄 分块 {i}:")
                print(f"       内容类型: 语义分块")
                print(f"       相似度评分: {chunk['similarity_score']:.4f}")
                print(f"      ⏰ 时间段: {time_range}")
                print(f"       文本内容: {chunk['text'][:120]}...")
            
            print(f"\n✅ 已完成第 {question_count} 个问题的处理")
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出问答系统")
            break
        except Exception as e:
            print(f"\n❌ 处理问题时出错: {e}")
            continue


def interactive_qa_mode():
    """交互式问答模式"""
    print("\n" + "="*80)
    print("🚀 交互式问答体验")
    print("="*80)
    
    video_data = get_mock_video_data()
    
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
• 使用RAGFlow文档处理框架
• 基于Qwen大语言模型生成回答

📝 问题示例:
• "糖尿病的主要症状有哪些?"
• "如何预防糖尿病的发生?"
• "糖尿病患者的饮食注意事项"
• "适合糖尿病患者的运动方式"
                """)
                continue
            
            # 处理查询
            print(f"\n🔄 正在思考您的问题...")
            time.sleep(1)  # 模拟处理时间
            print(f"    基于视频: {video_data['title']}")
            
            # 生成回答
            answer = generate_video_answer(user_input, video_data)
            
            # 显示结果
            print(f"\n🤖 智能回答:")
            print("═" * 60)
            print(answer)
            print("═" * 60)
            
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
        if mode == "metadata":
            # 仅演示元数据
            demo_video_metadata()
            
        elif mode == "prompt":
            # 仅演示提示词工程
            demo_prompt_engineering()
            
        elif mode == "qa":
            # 仅演示问答系统
            demo_intelligent_qa()
            
        elif mode == "interactive":
            # 仅运行交互模式
            interactive_qa_mode()
            
        else:
            # 完整演示流程
            print("🚀 开始简单视频RAG演示...")
            
            # 1. 元数据演示
            demo_video_metadata()
            input("\n⏳ 按Enter键继续...")
            
            # 2. 提示词工程演示
            demo_prompt_engineering()
            input("\n⏳ 按Enter键继续...")
            
            # 3. 智能问答演示
            demo_intelligent_qa()
            
            # 4. 交互模式选择
            user_choice = input("\n🤔 是否进入交互式问答体验? (y/n): ").strip().lower()
            if user_choice in ['y', 'yes', '是']:
                interactive_qa_mode()
    
    except Exception as e:
        print(f"\n❌ 系统运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 简单视频RAG演示完成!")


if __name__ == "__main__":
    main() 