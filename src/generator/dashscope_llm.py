"""


DashScope API
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx

from config.settings import get_settings
from ..utils.logger import get_logger
from ..api.schemas import GenerationContext, RetrievalResult

# 
settings = get_settings()
logger = get_logger(__name__)


class DashScopeLLM:
    """"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.api_key = self.settings.llm.api_key
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.model = self.settings.llm.model or "qwen-turbo"
        
        if not self.api_key:
            raise ValueError("APILLM_API_KEY")
    
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """"""
        
        try:
            self.logger.info(f": {len(prompt)}")
            
            # 
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # 
            payload = {
                "model": self.model,
                "input": {
                    "messages": messages
                },
                "parameters": {
                    "max_tokens": max_tokens or self.settings.llm.max_tokens,
                    "temperature": temperature or self.settings.llm.temperature,
                    "top_p": self.settings.llm.top_p,
                    "result_format": "message"
                }
            }
            
            # 
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # 
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code != 200:
                    error_msg = f"DashScope API: {response.status_code} - {response.text}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                
                # 
                result = response.json()
                
                if result.get("status_code") != 200:
                    error_msg = f"DashScope API: {result.get('message', '')}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                
                # 
                output = result.get("output", {})
                choices = output.get("choices", [])
                
                if not choices:
                    raise Exception("DashScope API")
                
                generated_text = choices[0].get("message", {}).get("content", "")
                
                self.logger.info(f": {len(generated_text)}")
                return generated_text
                
        except Exception as e:
            self.logger.error(f"DashScope: {e}")
            raise
    
    async def generate_answer_with_context(
        self,
        query: str,
        context: GenerationContext
    ) -> str:
        """"""
        
        try:
            self.logger.info(f": {query[:50]}...")
            
            # 
            system_prompt = """AI


1. 
2. 
3. 
4. 
5. 


- 
- 
- 
- 
- """

            # 
            context_parts = []
            
            # 
            text_contexts = [
                chunk for chunk in context.retrieved_chunks 
                if chunk.chunk_data.chunk_type.value in ["text_summary", "text_detail"]
            ]
            
            if text_contexts:
                context_parts.append("===  ===")
                for i, chunk in enumerate(text_contexts, 1):
                    chunk_data = chunk.chunk_data
                    context_link = chunk_data.metadata.context_link
                    context_parts.append(
                        f"{i}{context_link.chapter_number} {context_link.chapter_title}{context_link.page_number}\n"
                        f"{chunk_data.content}\n"
                    )
            
            # 
            media_contexts = [
                chunk for chunk in context.retrieved_chunks 
                if chunk.chunk_data.chunk_type.value in ["image", "video_clip"]
            ]
            
            if media_contexts:
                context_parts.append("\n===  ===")
                for i, chunk in enumerate(media_contexts, 1):
                    chunk_data = chunk.chunk_data
                    context_link = chunk_data.metadata.context_link
                    media_type = "" if chunk_data.chunk_type.value == "image" else ""
                    context_parts.append(
                        f"{media_type}{i}{context_link.chapter_number} {context_link.chapter_title}{context_link.page_number}\n"
                        f"{chunk_data.content}\n"
                        f"{chunk_data.metadata.source_url}\n"
                    )
            
            # 
            full_context = "\n".join(context_parts)
            
            prompt = f"""

{full_context}

{query}

"""

            # 
            answer = await self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.settings.llm.max_tokens,
                temperature=0.1  # 
            )
            
            return answer
            
        except Exception as e:
            self.logger.error(f": {e}")
            raise
    
    async def summarize_chapter(self, chapter_content: str, chapter_title: str) -> str:
        """"""
        
        try:
            system_prompt = """


1. 200-300
2. 
3. 
4. 
5. """

            prompt = f"""

{chapter_title}


{chapter_content}

"""

            summary = await self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f": {e}")
            return f"{str(e)}"
    
    async def generate_image_caption(self, image_description: str, context_info: str = "") -> str:
        """"""
        
        try:
            system_prompt = """


1. 
2. 
3. 
4. 100-200
5. """

            prompt = f"""

{image_description}

{f"{context_info}" if context_info else ""}

"""

            caption = await self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=300,
                temperature=0.4
            )
            
            return caption
            
        except Exception as e:
            self.logger.error(f": {e}")
            return f"{str(e)}"
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """"""
        
        try:
            system_prompt = """


1. 
2. 
3. 
4. 
5. """

            prompt = f"""{max_keywords}


{text}

"""

            result = await self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=200,
                temperature=0.2
            )
            
            # 
            keywords = [kw.strip() for kw in result.split(",") if kw.strip()]
            return keywords[:max_keywords]
            
        except Exception as e:
            self.logger.error(f": {e}")
            return []


# DashScope
_dashscope_llm: Optional[DashScopeLLM] = None


def get_dashscope_llm() -> DashScopeLLM:
    """DashScope LLM"""
    global _dashscope_llm
    
    if _dashscope_llm is None:
        _dashscope_llm = DashScopeLLM()
    
    return _dashscope_llm 