#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频元数据Schema定义
为RAGFlow系统提供统一的视频数据结构
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class VideoLanguage(Enum):
    """支持的视频语言类型"""
    CHINESE = "zh-CN"
    ENGLISH = "en-US"
    JAPANESE = "ja-JP"
    KOREAN = "ko-KR"


@dataclass
class SemanticChunk:
    """语义分块数据结构"""
    start_time_seconds: int
    end_time_seconds: int
    text: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "text": self.text
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticChunk':
        return cls(
            start_time_seconds=data["start_time_seconds"],
            end_time_seconds=data["end_time_seconds"],
            text=data["text"]
        )
    
    def duration(self) -> int:
        """获取分块时长（秒）"""
        return self.end_time_seconds - self.start_time_seconds
    
    def time_format(self) -> str:
        """格式化时间显示（MM:SS - MM:SS）"""
        start_min = self.start_time_seconds // 60
        start_sec = self.start_time_seconds % 60
        end_min = self.end_time_seconds // 60
        end_sec = self.end_time_seconds % 60
        return f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"


@dataclass
class VideoMetadata:
    """视频元数据完整结构"""
    video_id: str
    video_url: str
    title: str
    channel_name: str
    language: VideoLanguage
    ai_summary: str
    keywords: List[str]
    semantic_chunks: List[SemanticChunk]
    
    # 可选字段
    duration_seconds: Optional[int] = None
    thumbnail_url: Optional[str] = None
    upload_date: Optional[str] = None
    view_count: Optional[int] = None
    category: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "video_id": self.video_id,
            "video_url": self.video_url,
            "title": self.title,
            "channel_name": self.channel_name,
            "language": self.language.value,
            "ai_summary": self.ai_summary,
            "keywords": self.keywords,
            "semantic_chunks": [chunk.to_dict() for chunk in self.semantic_chunks],
            "duration_seconds": self.duration_seconds,
            "thumbnail_url": self.thumbnail_url,
            "upload_date": self.upload_date,
            "view_count": self.view_count,
            "category": self.category
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoMetadata':
        """从字典创建对象"""
        return cls(
            video_id=data["video_id"],
            video_url=data["video_url"],
            title=data["title"],
            channel_name=data["channel_name"],
            language=VideoLanguage(data["language"]),
            ai_summary=data["ai_summary"],
            keywords=data["keywords"],
            semantic_chunks=[SemanticChunk.from_dict(chunk) for chunk in data["semantic_chunks"]],
            duration_seconds=data.get("duration_seconds"),
            thumbnail_url=data.get("thumbnail_url"),
            upload_date=data.get("upload_date"),
            view_count=data.get("view_count"),
            category=data.get("category")
        )
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'VideoMetadata':
        """从JSON字符串创建对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_vectorizable_content(self) -> List[str]:
        """获取可向量化的文本内容"""
        content = [
            self.ai_summary,
            " ".join(self.keywords),
            self.title
        ]
        
        # 添加所有语义分块
        for chunk in self.semantic_chunks:
            content.append(chunk.text)
        
        return content
    
    def get_full_text(self) -> str:
        """获取所有文本内容的合并版本"""
        return " ".join(self.get_vectorizable_content())
    
    def get_chunk_by_time(self, time_seconds: int) -> Optional[SemanticChunk]:
        """根据时间获取对应的语义分块"""
        for chunk in self.semantic_chunks:
            if chunk.start_time_seconds <= time_seconds <= chunk.end_time_seconds:
                return chunk
        return None
    
    def search_chunks_by_keyword(self, keyword: str) -> List[SemanticChunk]:
        """根据关键词搜索相关的语义分块"""
        matching_chunks = []
        keyword_lower = keyword.lower()
        
        for chunk in self.semantic_chunks:
            if keyword_lower in chunk.text.lower():
                matching_chunks.append(chunk)
        
        return matching_chunks


# 示例糖尿病视频数据
EXAMPLE_DIABETES_VIDEO = VideoMetadata(
    video_id="au-w0QXB6jg",
    video_url="https://www.youtube.com/watch?v=au-w0QXB6jg",
    title="了解2型糖尿病 (Understanding Type 2 Diabetes Mellitus)",
    channel_name="Nucleus Medical Media",
    language=VideoLanguage.CHINESE,
    duration_seconds=150,
    category="健康医疗",
    
    ai_summary="这个视频详细介绍了2型糖尿病的基本概念、发病机制、主要症状以及预防和管理方法。内容涵盖了胰岛素抵抗、血糖控制、饮食管理、运动疗法等关键知识点，为糖尿病患者和高危人群提供了实用的健康指导。",
    
    keywords=[
        "2型糖尿病", "Type 2 Diabetes", "血糖控制", "胰岛素抵抗", "饮食管理",
        "运动疗法", "健康生活", "预防措施", "症状识别", "医疗指导"
    ],
    
    semantic_chunks=[
        SemanticChunk(
            start_time_seconds=15,
            end_time_seconds=45,
            text="2型糖尿病是一种慢性代谢疾病，主要特征是身体无法有效利用胰岛素或产生足够的胰岛素来维持正常的血糖水平。这种疾病通常在成年后发病，与生活方式、遗传因素和年龄有密切关系。"
        ),
        SemanticChunk(
            start_time_seconds=46,
            end_time_seconds=75,
            text="2型糖尿病的主要症状包括频繁口渴、多尿、疲劳、视力模糊、伤口愈合缓慢等。如果不及时诊断和治疗，可能会导致严重的并发症，如心血管疾病、肾病、神经病变和眼部疾病。"
        ),
        SemanticChunk(
            start_time_seconds=76,
            end_time_seconds=100,
            text="饮食管理是2型糖尿病治疗的关键组成部分。患者应该选择低糖、高纤维的食物，控制碳水化合物的摄入量，保持规律的进餐时间。建议多吃蔬菜、瘦肉、全谷物，避免加工食品和含糖饮料。"
        ),
        SemanticChunk(
            start_time_seconds=101,
            end_time_seconds=130,
            text="规律的体育锻炼对2型糖尿病患者非常重要。适量的有氧运动和力量训练可以帮助改善胰岛素敏感性，降低血糖水平，控制体重。建议每周至少进行150分钟的中等强度运动，如快走、游泳或骑自行车。"
        )
    ]
)


def create_video_metadata_template() -> str:
    """创建视频元数据模板"""
    template = {
        "video_id": "string",
        "video_url": "string",
        "title": "string",
        "channel_name": "string",
        "language": "zh-CN",
        "ai_summary": "string",
        "keywords": ["string"],
        "semantic_chunks": [
            {
                "start_time_seconds": "integer",
                "end_time_seconds": "integer",
                "text": "string"
            }
        ],
        "duration_seconds": "integer (optional)",
        "thumbnail_url": "string (optional)",
        "upload_date": "string (optional)",
        "view_count": "integer (optional)",
        "category": "string (optional)"
    }
    
    return json.dumps(template, ensure_ascii=False, indent=2)


def display_video_metadata_detailed(video: VideoMetadata, show_full_content: bool = True):
    """详细显示视频元数据信息"""
    print("📺 视频详细信息")
    print("=" * 80)
    
    # 基本信息
    print("📋 基本信息:")
    print(f"    视频ID: {video.video_id}")
    print(f"    视频URL: {video.video_url}")
    print(f"    标题: {video.title}")
    print(f"    频道: {video.channel_name}")
    print(f"     语言: {video.language.value}")
    print(f"     分类: {video.category}")
    print(f"   ⏱ 时长: {video.duration_seconds}秒")
    
    # 可选信息
    if video.thumbnail_url:
        print(f"     缩略图: {video.thumbnail_url}")
    if video.upload_date:
        print(f"    上传日期: {video.upload_date}")
    if video.view_count:
        print(f"     观看次数: {video.view_count:,}")
    
    # AI生成摘要
    print(f"\n🤖 AI生成摘要:")
    if show_full_content:
        print(f"   {video.ai_summary}")
    else:
        print(f"   {video.ai_summary[:200]}...")
    print(f"    字符数: {len(video.ai_summary)} 个")
    
    # 关键词
    print(f"\n🏷 关键词标签 ({len(video.keywords)}个):")
    for i, keyword in enumerate(video.keywords, 1):
        print(f"   {i:2d}. {keyword}")
    print(f"    总字符数: {len(' '.join(video.keywords))} 个")
    
    # 语义分块
    print(f"\n📝 语义分块 ({len(video.semantic_chunks)}个):")
    total_chunk_length = 0
    
    for i, chunk in enumerate(video.semantic_chunks, 1):
        duration = chunk.duration()
        total_chunk_length += len(chunk.text)
        
        print(f"\n     分块 {i}:")
        print(f"      ⏰ 时间段: {chunk.time_format()} (时长: {duration}秒)")
        print(f"       字符数: {len(chunk.text)} 个")
        
        if show_full_content:
            # 显示完整内容
            print(f"       内容:")
            print(f"         {chunk.text}")
        else:
            # 只显示前100个字符
            print(f"       内容预览: {chunk.text[:100]}...")
    
    print(f"\n📊 分块统计:")
    print(f"    分块总数: {len(video.semantic_chunks)}个")
    print(f"    总字符数: {total_chunk_length:,} 个")
    print(f"    平均长度: {total_chunk_length // len(video.semantic_chunks):,} 个字符")
    
    # 可向量化内容统计
    vectorizable_content = video.get_vectorizable_content()
    print(f"\n🔍 可向量化内容:")
    print(f"    内容片段数: {len(vectorizable_content)}个")
    
    for i, content in enumerate(vectorizable_content, 1):
        content_type = ""
        if i == 1:
            content_type = "AI摘要"
        elif i == 2:
            content_type = "关键词"
        elif i == 3:
            content_type = "标题"
        else:
            content_type = f"分块{i-3}"
        
        print(f"   {i:2d}. {content_type}: {len(content):,} 个字符")
        if show_full_content and len(content) < 200:
            print(f"       完整内容: {content}")
        else:
            print(f"       内容预览: {content[:100]}...")
    
    total_vectorizable_length = sum(len(content) for content in vectorizable_content)
    print(f"    总字符数: {total_vectorizable_length:,} 个")


if __name__ == "__main__":
    # 演示视频元数据Schema
    print("📊 视频元数据Schema演示")
    print("=" * 80)
    
    # 1. 显示详细信息
    display_video_metadata_detailed(EXAMPLE_DIABETES_VIDEO, show_full_content=True)
    
    # 2. JSON格式输出
    print(f"\n" + "="*80)
    print("📄 JSON格式输出:")
    print("="*80)
    print(EXAMPLE_DIABETES_VIDEO.to_json())
    
    # 3. 功能测试
    print(f"\n" + "="*80)
    print("🧪 功能测试:")
    print("="*80)
    
    # 按时间查找分块
    test_time = 50
    chunk = EXAMPLE_DIABETES_VIDEO.get_chunk_by_time(test_time)
    if chunk:
        print(f"⏰ 时间点 {test_time}秒 对应的分块:")
        print(f"   时间段: {chunk.time_format()}")
        print(f"   内容预览: {chunk.text[:100]}...")
    
    # 按关键词搜索
    test_keyword = "糖尿病"
    matching_chunks = EXAMPLE_DIABETES_VIDEO.search_chunks_by_keyword(test_keyword)
    print(f"\n🔍 包含关键词 '{test_keyword}' 的分块:")
    for i, chunk in enumerate(matching_chunks, 1):
        print(f"   {i}. {chunk.time_format()}: {chunk.text[:80]}...")
    
    # 统计信息
    print(f"\n📈 统计信息:")
    print(f"    视频总时长: {EXAMPLE_DIABETES_VIDEO.duration_seconds}秒 ({EXAMPLE_DIABETES_VIDEO.duration_seconds//60}分{EXAMPLE_DIABETES_VIDEO.duration_seconds%60}秒)")
    print(f"    内容覆盖时长: {EXAMPLE_DIABETES_VIDEO.semantic_chunks[-1].end_time_seconds - EXAMPLE_DIABETES_VIDEO.semantic_chunks[0].start_time_seconds}秒")
    print(f"    全文字符数: {len(EXAMPLE_DIABETES_VIDEO.get_full_text()):,} 个")
    print(f"     关键词数量: {len(EXAMPLE_DIABETES_VIDEO.keywords)} 个")
    print(f"    平均分块时长: {sum(chunk.duration() for chunk in EXAMPLE_DIABETES_VIDEO.semantic_chunks) // len(EXAMPLE_DIABETES_VIDEO.semantic_chunks)}秒") 