#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG - 
Milvus
"""

import sys
import time
import os
import dashscope
from pathlib import Path

# Python
sys.path.insert(0, str(Path(__file__).parent.parent))

# 
from ragflow_integration.video_metadata_schema import VideoMetadata, SemanticChunk, VideoLanguage, EXAMPLE_DIABETES_VIDEO, display_video_metadata_detailed

# API
os.environ['DASHSCOPE_API_KEY'] = 'sk-b70842d25c884aa9aa18955b00c24d37'
dashscope.api_key = 'sk-b70842d25c884aa9aa18955b00c24d37'


def print_banner():
    """"""
    banner = """

                    RAG                                 
                                                                              
   :                                                                
     •                                                    
     • RAGFlow                                                    
     • Qwen +                                           
     •                                                  
                                                                              
   :  +  +                               

"""
    print(banner)


def get_mock_video_data():
    """"""
    return {
        "video_url": "https://www.youtube.com/watch?v=au-w0QXB6jg",
        "title": " 2  (Understanding Type 2 Diabetes Mellitus)",
        "channel_name": "Nucleus Medical Media",
        "category": "",
        "ai_summary": "22",
        "keywords": "2, , , , , ",
        "semantic_chunks": [
            {
                "start_time_seconds": 46,
                "end_time_seconds": 75,
                "text": "2''''",
                "similarity_score": 0.8924
            },
            {
                "start_time_seconds": 76,
                "end_time_seconds": 100,
                "text": "",
                "similarity_score": 0.8456
            },
            {
                "start_time_seconds": 101,
                "end_time_seconds": 130,
                "text": "2",
                "similarity_score": 0.7892
            }
        ]
    }


def build_qwen_prompt(query: str, video_context: dict) -> str:
    """Qwen"""
    
    # 
    semantic_chunks_text = ""
    for i, chunk in enumerate(video_context.get("semantic_chunks", []), 1):
        start_min = chunk["start_time_seconds"] // 60
        start_sec = chunk["start_time_seconds"] % 60
        end_min = chunk["end_time_seconds"] // 60
        end_sec = chunk["end_time_seconds"] % 60
        
        semantic_chunks_text += f'''
    {{
      "": "{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}",
      "": {chunk["similarity_score"]:.4f},
      "": "{chunk["text"]}"
    }}'''
        if i < len(video_context["semantic_chunks"]):
            semantic_chunks_text += ","
    
    prompt = f""" (Role):
AI

 (Context):
"{video_context.get('title', '')}"

[]
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

 (Task & Instructions):


1. : []ai_summarysemantic_chunks

2. : 

3. : video_url""

4. : ""

 (User's Query):
"{query}"

"""

    return prompt


def generate_video_answer(query: str, video_context: dict) -> str:
    """"""
    try:
        print(" ...")
        
        # 
        prompt = build_qwen_prompt(query, video_context)
        
        print(f" : {len(prompt)} ")
        
        # Qwen
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
            print(f" : {generate_time:.2f}")
            return answer
        else:
            print(f" Qwen: {response}")
            return ""
            
    except Exception as e:
        print(f" : {e}")
        return f": {str(e)}"


def demo_video_metadata():
    """"""
    print("\n" + "="*80)
    print(" ")
    print("="*80)
    
    video_data = get_mock_video_data()
    
    print(" :")
    print("-" * 60)
    print(f" : {video_data['title']}")
    print(f" : {video_data['video_url']}")
    print(f" : {video_data['channel_name']}")
    print(f"  : {video_data['category']}")
    
    print(f"\n AI:")
    print(f"   {video_data['ai_summary']}")
    
    print(f"\n  :")
    print(f"   {video_data['keywords']}")
    
    print(f"\n  ({len(video_data['semantic_chunks'])}):")
    for i, chunk in enumerate(video_data['semantic_chunks'], 1):
        start_min = chunk["start_time_seconds"] // 60
        start_sec = chunk["start_time_seconds"] % 60
        end_min = chunk["end_time_seconds"] // 60
        end_sec = chunk["end_time_seconds"] % 60
        time_range = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
        
        print(f"   {i}. : {time_range} | : {chunk['similarity_score']:.4f}")
        print(f"      : {chunk['text']}")
        print()


def demo_prompt_engineering():
    """"""
    print("\n" + "="*80)
    print(" RAGFlow")
    print("="*80)
    
    print(" :")
    print("-" * 60)
    
    structure = """
 1.  (Role):
   • AI
   • 
   • 

 2.  (Context):
   • 
   • AI + 
   •  + 

 3.  (Instructions):
   • 
   • 
   • 
   • 

 4.  (Query):
   • 
   • 
"""
    print(structure)
    
    print(" :")
    print("     - JSON")
    print("     - AI++")
    print("     - ")
    print("     - 4")
    print("     - ")
    print("     - ")


def demo_intelligent_qa():
    """"""
    print("\n" + "="*80)
    print(" ")
    print("="*80)
    
    video_data = get_mock_video_data()
    
    print(" !")
    print(" :  'quit'  'help' ")
    print(" ")
    
    question_count = 0
    
    while True:
        try:
            user_input = input(f"\n  ({question_count + 1}): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '', 'q']:
                print(" !")
                break
            
            if user_input.lower() in ['help', '', 'h']:
                print("""
 :
• YouTube
• RAGFlow
• Qwen + 

 :
• "2"
• ""
• "2"
• "2"
                """)
                continue
            
            question_count += 1
            print(f"\n {question_count}...")
            print("="*60)
            
            # 
            print(" ...")
            print(f"    : {video_data['title']}")
            print(f"    : {max([chunk['similarity_score'] for chunk in video_data['semantic_chunks']]):.4f}")
            print(f"    : {len(video_data['semantic_chunks'])}")
            
            # 
            answer = generate_video_answer(user_input, video_data)
            
            print(f"\n AI:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
            # 
            print(f"\n" + "="*80)
            print(" ")
            print("="*80)
            
            # 
            print(" :")
            display_video_metadata_detailed(EXAMPLE_DIABETES_VIDEO, show_full_content=False)
            
            print(f"\n :")
            for i, chunk in enumerate(video_data['semantic_chunks'], 1):
                start_min = chunk["start_time_seconds"] // 60
                start_sec = chunk["start_time_seconds"] % 60
                end_min = chunk["end_time_seconds"] // 60
                end_sec = chunk["end_time_seconds"] % 60
                time_range = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
                
                print(f"\n     {i}:")
                print(f"       : ")
                print(f"       : {chunk['similarity_score']:.4f}")
                print(f"      ⏰ : {time_range}")
                print(f"       : {chunk['text'][:120]}...")
            
            print(f"\n :  {question_count} ")
            
        except KeyboardInterrupt:
            print("\n\n ")
            break
        except Exception as e:
            print(f"\n : {e}")
            continue


def interactive_qa_mode():
    """"""
    print("\n" + "="*80)
    print(" ")
    print("="*80)
    
    video_data = get_mock_video_data()
    
    print(" !")
    print(" :  'quit'  'help' ")
    
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
• YouTube
• RAGFlow
• Qwen + 

 :
• "2"
• ""
• ""
• "2"
                """)
                continue
            
            # 
            print(f"\n ...")
            time.sleep(1)  # 
            print(f"    : {video_data['title']}")
            
            # 
            answer = generate_video_answer(user_input, video_data)
            
            # 
            print(f"\n :")
            print("" * 60)
            print(answer)
            print("" * 60)
            
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
        if mode == "metadata":
            # 
            demo_video_metadata()
            
        elif mode == "prompt":
            # 
            demo_prompt_engineering()
            
        elif mode == "qa":
            # 
            demo_intelligent_qa()
            
        elif mode == "interactive":
            # 
            interactive_qa_mode()
            
        else:
            # 
            print(" RAG...")
            
            # 1. 
            demo_video_metadata()
            input("\n⏳ Enter...")
            
            # 2. 
            demo_prompt_engineering()
            input("\n⏳ Enter...")
            
            # 3. 
            demo_intelligent_qa()
            
            # 4. 
            user_choice = input("\n ? (y/n): ").strip().lower()
            if user_choice in ['y', 'yes', '']:
                interactive_qa_mode()
    
    except Exception as e:
        print(f"\n : {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n RAG!")


if __name__ == "__main__":
    main() 