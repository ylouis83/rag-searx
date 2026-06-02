#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强视频RAG管道 - 多路召回系统
基于多种检索策略的智能视频问答系统
"""

import os
import sys
import time
import numpy as np
import dashscope
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragflow_integration.video_metadata_schema import VideoMetadata, SemanticChunk
from ragflow_integration.bge_embedding import BGEEmbeddingModel
from ragflow_integration.multi_path_recall import MultiPathRecallSystem, MultiRecallConfig, RecallResult, RecallPath

# 设置通义千问API
os.environ['DASHSCOPE_API_KEY'] = os.environ.get("DASHSCOPE_API_KEY", "")
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")


class EnhancedVideoRAGPipeline:
    """增强视频RAG管道 - 支持多路召回"""
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-zh-v1.5",
        recall_config: MultiRecallConfig = None
    ):
        """
        初始化增强RAG管道
        
        Args:
            embedding_model_name: BGE向量模型名称
            recall_config: 多路召回配置
        """
        
        self.embedding_model_name = embedding_model_name
        self.recall_config = recall_config or MultiRecallConfig(
            vector_weight=0.4,
            keyword_weight=0.4,
            semantic_weight=0.2,
            vector_top_k=15,
            keyword_top_k=10,
            semantic_top_k=8,
            final_top_k=10
        )
        
        # 初始化系统组件
        print("🚀 正在初始化增强RAG管道...")
        self._init_components()
        
        # 存储数据
        self.videos_metadata = []
        self.indexed_documents = []
        
        print("✅ 增强RAG管道初始化完成")
    
    def _init_components(self):
        """初始化系统组件"""
        
        # 1. 初始化BGE向量模型
        print("🔧 正在初始化BGE向量模型...")
        self.embedding_model = BGEEmbeddingModel(
            model_name=self.embedding_model_name,
            max_seq_length=512
        )
        
        # 2. 初始化多路召回系统
        print("🛠 正在初始化多路召回系统...")
        self.recall_system = MultiPathRecallSystem(self.recall_config)
        self.recall_system.set_embedding_model(self.embedding_model)
        
        print("✅ 系统组件初始化完成")
    
    def store_video_metadata(self, video_metadata: VideoMetadata) -> bool:
        """
        存储视频元数据并建立索引
        
        Args:
            video_metadata: 视频元数据对象
        
        Returns:
            存储是否成功
        """
        
        try:
            print(f"💾 正在存储视频: {video_metadata.title}")
            
            # 存储元数据
            self.videos_metadata.append(video_metadata)
            
            # 构建文档
            documents = self._build_documents_from_video(video_metadata)
            self.indexed_documents.extend(documents)
            
            # 建立多路索引
            self.recall_system.build_index(self.indexed_documents)
            
            print(f"✅ 视频存储完成")
            print(f"    视频ID: {video_metadata.video_id}")
            print(f"    视频标题: {video_metadata.title}")
            print(f"    语义分块: {len(video_metadata.semantic_chunks)} 个")
            print(f"    文档条目: {len(documents)} 个")
            
            return True
            
        except Exception as e:
            print(f"❌ 视频存储失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _build_documents_from_video(self, video_metadata: VideoMetadata) -> List[Dict[str, Any]]:
        """
        从视频元数据构建文档列表
        
        Args:
            video_metadata: 视频元数据
        
        Returns:
            文档列表
        """
        
        documents = []
        
        # 1. AI生成摘要文档
        documents.append({
            "id": f"{video_metadata.video_id}_summary",
            "content": video_metadata.ai_summary,
            "metadata": {
                "video_id": video_metadata.video_id,
                "video_url": video_metadata.video_url,
                "title": video_metadata.title,
                "channel_name": video_metadata.channel_name,
                "category": video_metadata.category,
                "content_type": "summary",
                "duration_seconds": video_metadata.duration_seconds,
                "language": video_metadata.language.value,
                "keywords": " ".join(video_metadata.keywords)
            }
        })
        
        # 2. 关键词文档
        keywords_text = " ".join(video_metadata.keywords)
        documents.append({
            "id": f"{video_metadata.video_id}_keywords",
            "content": keywords_text,
            "metadata": {
                "video_id": video_metadata.video_id,
                "video_url": video_metadata.video_url,
                "title": video_metadata.title,
                "channel_name": video_metadata.channel_name,
                "category": video_metadata.category,
                "content_type": "keywords",
                "duration_seconds": video_metadata.duration_seconds,
                "language": video_metadata.language.value,
                "keywords": keywords_text
            }
        })
        
        # 3. 语义分块文档
        for i, chunk in enumerate(video_metadata.semantic_chunks):
            documents.append({
                "id": f"{video_metadata.video_id}_chunk_{i}",
                "content": chunk.text,
                "metadata": {
                    "video_id": video_metadata.video_id,
                    "video_url": video_metadata.video_url,
                    "title": video_metadata.title,
                    "channel_name": video_metadata.channel_name,
                    "category": video_metadata.category,
                    "content_type": "chunk",
                    "chunk_start_time": chunk.start_time_seconds,
                    "chunk_end_time": chunk.end_time_seconds,
                    "chunk_index": i,
                    "duration_seconds": video_metadata.duration_seconds,
                    "language": video_metadata.language.value,
                    "keywords": " ".join(video_metadata.keywords)
                }
            })
        
        return documents
    
    def enhanced_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        执行多路召回搜索 - 融合向量、关键词、语义检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            搜索结果列表
        """
        
        print(f"🔍 执行多路召回搜索: {query}")
        start_time = time.time()
        
        if not self.indexed_documents:
            print("⚠️ 暂无索引文档")
            return []
        
        # 执行多路召回
        recall_results = self.recall_system.multi_recall(query, top_k)
        
        # 构建搜索结果
        search_results = []
        for result in recall_results:
            # 找到对应的文档
            doc = next((doc for doc in self.indexed_documents if doc["id"] == result.doc_id), None)
            if doc:
                search_result = {
                    "id": result.doc_id,
                    "score": result.score,
                    "content": result.content,
                    "recall_path": result.recall_path.value,
                    **doc["metadata"]
                }
                search_results.append(search_result)
        
        search_time = time.time() - start_time
        print(f"✅ 多路召回完成: {search_time:.3f}秒，共找到 {len(search_results)} 个结果")
        
        return search_results
    
    def generate_enhanced_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        基于搜索结果生成增强回答
        
        Args:
            query: 用户查询
            search_results: 搜索结果
        
        Returns:
            生成的回答文本
        """
        
        if not search_results:
            return "抱歉，没有找到相关内容来回答您的问题。"
        
        try:
            print("🤖 正在生成智能回答...")
            
            # 构建增强上下文
            enhanced_context = self._build_enhanced_context(search_results)
            
            # 构建提示词
            prompt = self._build_enhanced_prompt(query, enhanced_context, search_results)
            
            print(f"📝 提示词长度: {len(prompt)} 个字符")
            
            # 调用通义千问生成回答
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
    
    def _build_enhanced_context(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建增强上下文信息"""
        
        # 按视频分组结果
        videos = {}
        recall_paths_used = set()
        
        for result in search_results:
            video_id = result.get("video_id", "")
            recall_paths_used.add(result.get("recall_path", ""))
            
            if video_id not in videos:
                videos[video_id] = {
                    "video_id": video_id,
                    "video_url": result.get("video_url", ""),
                    "title": result.get("title", ""),
                    "channel_name": result.get("channel_name", ""),
                    "category": result.get("category", ""),
                    "semantic_chunks": [],
                    "summaries": [],
                    "keywords": [],
                    "max_score": 0
                }
            
            videos[video_id]["max_score"] = max(videos[video_id]["max_score"], result.get("score", 0))
            
            # 按内容类型分类
            content_type = result.get("content_type", "")
            if content_type == "chunk":
                videos[video_id]["semantic_chunks"].append({
                    "start_time_seconds": result.get("chunk_start_time", 0),
                    "end_time_seconds": result.get("chunk_end_time", 0),
                    "text": result.get("content", ""),
                    "similarity_score": result.get("score", 0),
                    "recall_path": result.get("recall_path", "")
                })
            elif content_type == "summary":
                videos[video_id]["summaries"].append({
                    "text": result.get("content", ""),
                    "score": result.get("score", 0),
                    "recall_path": result.get("recall_path", "")
                })
            elif content_type == "keywords":
                videos[video_id]["keywords"].append({
                    "text": result.get("content", ""),
                    "score": result.get("score", 0),
                    "recall_path": result.get("recall_path", "")
                })
        
        # 选择最佳视频
        best_video = max(videos.values(), key=lambda x: x["max_score"]) if videos else {}
        
        return {
            "best_video": best_video,
            "recall_paths_used": list(recall_paths_used),
            "total_results": len(search_results),
            "videos_found": len(videos)
        }
    
    def _build_enhanced_prompt(self, query: str, context: Dict[str, Any], search_results: List[Dict[str, Any]]) -> str:
        """构建增强提示词"""
        
        best_video = context.get("best_video", {})
        recall_paths = context.get("recall_paths_used", [])
        
        # 召回路径说明
        recall_info = ""
        if "vector" in recall_paths:
            recall_info += "• 向量召回: 基于语义向量相似度的精确匹配\n"
        if "keyword" in recall_paths:
            recall_info += "• 关键词召回: 基于关键词匹配的文本检索\n"
        if "semantic" in recall_paths:
            recall_info += "• 语义召回: 基于深度语义理解的智能检索\n"
        if "hybrid" in recall_paths:
            recall_info += "• 混合召回: 多种策略融合的综合检索\n"
        
        # 构建分块信息
        chunks_text = ""
        chunks = best_video.get("semantic_chunks", [])[:3]  # 取前3个最相关的分块
        for i, chunk in enumerate(chunks, 1):
            start_min = chunk["start_time_seconds"] // 60
            start_sec = chunk["start_time_seconds"] % 60
            end_min = chunk["end_time_seconds"] // 60
            end_sec = chunk["end_time_seconds"] % 60
            
            chunks_text += f'''
    {{
      "分块{i}": {{
        "时间段": "{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}",
        "相似度": {chunk["similarity_score"]:.4f},
        "召回路径": "{chunk.get("recall_path", "unknown")}",
        "内容": "{chunk["text"]}"
      }}
    }},'''
        
        chunks_text = chunks_text.rstrip(",")
        
        # 构建摘要信息
        summaries_text = ""
        summaries = best_video.get("summaries", [])
        if summaries:
            best_summary = max(summaries, key=lambda x: x["score"])
            summaries_text = f'"ai_summary": "{best_summary["text"]}",'
        
        # 构建关键词信息
        keywords_text = ""
        keywords = best_video.get("keywords", [])
        if keywords:
            best_keywords = max(keywords, key=lambda x: x["score"])
            keywords_text = f'"keywords": "{best_keywords["text"]}",'
        
        prompt = f"""角色定义 (Role):
你是一个专业的AI视频内容助手，基于YouTube视频内容为用户提供准确、有用的回答。

多路召回信息 (Multi-Path Recall):
本次检索使用了以下召回策略：
{recall_info}
共检索到 {context.get("total_results", 0)} 个相关结果，来自 {context.get("videos_found", 0)} 个视频

上下文信息 (Context):
基于多路召回系统，以下是最相关的视频内容：

[最相关视频信息]
{{
  "video_url": "{best_video.get('video_url', '')}",
  "title": "{best_video.get('title', '')}",
  "channel_name": "{best_video.get('channel_name', '')}",
  "category": "{best_video.get('category', '')}",
  {summaries_text}
  {keywords_text}
  "semantic_chunks": [{chunks_text}
  ]
}}

增强指令 (Enhanced Instructions):
请基于以上通过多路召回获得的视频内容信息，为用户提供准确、详细的回答：

1. 优先使用相似度最高的语义分块内容来回答问题

2. 如果需要，可以结合AI摘要和关键词信息提供更全面的答案

3. 在回答中自然地提及相关的时间段信息，方便用户定位到具体内容

4. 如果信息不足以完全回答问题，请诚实说明，并建议用户观看完整视频: "{best_video.get('video_url', '')}"

5. 保持回答简洁明了，重点突出，避免冗余信息

用户问题 (User Query):
"{query}"

请基于以上信息提供专业、准确的回答："""

        return prompt
    
    def enhanced_query(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """
        执行完整的增强查询 - 多路召回+智能生成
        
        Args:
            question: 用户问题
            top_k: 返回结果数量
        
        Returns:
            完整的查询结果
        """
        
        start_time = time.time()
        
        print(f"\n🚀 开始增强RAG查询")
        print(f"❓ 用户问题: {question}")
        print("="*80)
        
        # 1. 执行多路召回搜索
        search_results = self.enhanced_search(question, top_k)
        
        if not search_results:
            return {
                "question": question,
                "answer": "抱歉，没有找到相关内容来回答您的问题。请尝试使用不同的关键词或更具体的问题。",
                "search_results": [],
                "total_time": time.time() - start_time,
                "recall_paths": []
            }
        
        # 2. 生成智能回答
        answer = self.generate_enhanced_answer(question, search_results)
        
        total_time = time.time() - start_time
        
        # 3. 统计分析信息
        recall_paths = list(set([r.get("recall_path", "") for r in search_results]))
        path_stats = {}
        for path in recall_paths:
            path_results = [r for r in search_results if r.get("recall_path") == path]
            path_stats[path] = {
                "count": len(path_results),
                "avg_score": np.mean([r.get("score", 0) for r in path_results]),
                "max_score": max([r.get("score", 0) for r in path_results])
            }
        
        # 返回完整结果
        result = {
            "question": question,
            "answer": answer,
            "search_results": search_results,
            "total_time": total_time,
            "recall_paths": recall_paths,
            "path_statistics": path_stats,
            "stats": {
                "retrieved_videos": len(set([r.get("video_id", "") for r in search_results])),
                "retrieved_documents": len(search_results),
                "avg_similarity": np.mean([r.get("score", 0) for r in search_results]),
                "max_similarity": max([r.get("score", 0) for r in search_results]),
                "recall_diversity": len(recall_paths)
            }
        }
        
        print(f"✅ 增强RAG查询完成，总耗时: {total_time:.2f}秒")
        
        return result
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        
        recall_stats = self.recall_system.get_stats()
        
        return {
            "videos_stored": len(self.videos_metadata),
            "documents_indexed": len(self.indexed_documents),
            "embedding_model": self.embedding_model_name,
            "recall_system": recall_stats,
            "recall_config": {
                "vector_weight": self.recall_config.vector_weight,
                "keyword_weight": self.recall_config.keyword_weight,
                "semantic_weight": self.recall_config.semantic_weight,
                "final_top_k": self.recall_config.final_top_k
            }
        }


# 测试代码
if __name__ == "__main__":
    from ragflow_integration.video_metadata_schema import EXAMPLE_DIABETES_VIDEO
    
    print("🚀 增强视频RAG系统测试")
    print("=" * 50)
    
    # 创建增强配置
    enhanced_config = MultiRecallConfig(
        vector_weight=0.4,
        keyword_weight=0.4,
        semantic_weight=0.2,
        vector_top_k=15,
        keyword_top_k=10,
        semantic_top_k=8,
        final_top_k=8
    )
    
    # 初始化管道
    enhanced_pipeline = EnhancedVideoRAGPipeline(recall_config=enhanced_config)
    
    # 存储测试数据
    print("\n💾 正在存储测试视频数据...")
    success = enhanced_pipeline.store_video_metadata(EXAMPLE_DIABETES_VIDEO)
    
    if success:
        # 测试查询
        test_queries = [
            "糖尿病有哪些症状",
            "如何预防糖尿病",
            "糖尿病患者的饮食建议",
            "血糖控制的重要性"
        ]
        
        for query in test_queries:
            print(f"\n🔍 测试查询: {query}")
            result = enhanced_pipeline.enhanced_query(query, top_k=8)
            print(f"    检索结果: {len(result['search_results'])} 个")
            print(f"     召回路径: {', '.join(result['recall_paths'])}")
            print(f"    召回多样性: {result['stats']['recall_diversity']}分")
        
        # 系统统计
        print("\n📊 系统统计信息:")
        stats = enhanced_pipeline.get_system_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"     {k}: {v}")
            else:
                print(f"   {key}: {value}")
    else:
        print("❌ 测试失败，数据存储出错") 