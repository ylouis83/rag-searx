#!/usr/bin/env python3
"""
RAG-SearX 

TRD
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Python
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from src.utils.logger import get_logger

# 
settings = get_settings()
logger = get_logger(__name__)


class IndexBuilder:
    """"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        
    async def build_index(
        self, 
        book_path: str, 
        media_path: Optional[str] = None,
        book_title: Optional[str] = None,
        author: Optional[str] = None,
        language: str = "zh"
    ) -> bool:
        """
        
        
        Args:
            book_path: 
            media_path: 
            book_title: 
            author: 
            language: 
            
        Returns:
            
        """
        
        try:
            self.logger.info(f": {book_path}")
            start_time = time.time()
            
            # 1. 
            book_file = Path(book_path)
            if not book_file.exists():
                raise FileNotFoundError(f": {book_path}")
            
            # 2.  (FR-1.1)
            self.logger.info("1: ")
            document_data = await self._extract_document(book_file, book_title, author, language)
            
            media_files = []
            if media_path:
                media_files = await self._extract_media_files(media_path)
            
            # 3.  (FR-1.2)
            self.logger.info("2: ")
            text_chunks = await self._process_text(document_data)
            
            # 4.  (FR-1.3)
            self.logger.info("3: ")
            image_chunks = []
            for media_file in media_files:
                if self._is_image_file(media_file):
                    chunks = await self._process_images([media_file])
                    image_chunks.extend(chunks)
            
            # 5.  (FR-1.4)
            self.logger.info("4: ")
            video_chunks = []
            for media_file in media_files:
                if self._is_video_file(media_file):
                    chunks = await self._process_videos([media_file])
                    video_chunks.extend(chunks)
            
            # 6.  (FR-1.5)
            self.logger.info("5: ")
            all_chunks = text_chunks + image_chunks + video_chunks
            await self._store_vectors(all_chunks)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f": {elapsed_time:.2f}")
            self.logger.info(f" {len(all_chunks)} ")
            
            return True
            
        except Exception as e:
            self.logger.error(f": {e}")
            return False
    
    async def _extract_document(self, book_file: Path, title: Optional[str], author: Optional[str], language: str) -> dict:
        """"""
        
        self.logger.info(f": {book_file.name}")
        
        # TODO: 
        # PDFePubTXT
        
        return {
            "id": f"doc_{int(time.time())}",
            "title": title or book_file.stem,
            "author": author,
            "language": language,
            "file_path": str(book_file),
            "content": "...",  # 
            "chapters": [
                {
                    "id": f"chap_1",
                    "number": 1,
                    "title": "",
                    "content": "...",
                    "start_page": 1,
                    "end_page": 10,
                }
            ]
        }
    
    async def _extract_media_files(self, media_path: str) -> List[Path]:
        """"""
        
        media_dir = Path(media_path)
        if not media_dir.exists():
            self.logger.warning(f": {media_path}")
            return []
        
        media_files = []
        
        # 
        image_extensions = settings.media.image_formats_list
        video_extensions = settings.media.video_formats_list
        
        for file_path in media_dir.rglob("*"):
            if file_path.is_file():
                extension = file_path.suffix.lower().lstrip('.')
                if extension in image_extensions + video_extensions:
                    media_files.append(file_path)
        
        self.logger.info(f" {len(media_files)} ")
        return media_files
    
    async def _process_text(self, document_data: dict) -> List[dict]:
        """"""
        
        self.logger.info("")
        
        chunks = []
        
        # 
        for chapter in document_data.get("chapters", []):
            # 
            summary_chunk = {
                "id": f"chunk_summary_{chapter['id']}",
                "content": f": {chapter['title']} - ...",
                "type": "text_summary",
                "chapter_id": chapter["id"],
                "metadata": {
                    "media_type": "text_summary",
                    "book_title": document_data["title"],
                    "source_text": f"{chapter['number']}",
                    "context_link": {
                        "chapter_number": chapter["number"],
                        "chapter_title": chapter["title"],
                        "page_number": chapter["start_page"],
                    }
                }
            }
            chunks.append(summary_chunk)
            
            # 
            detail_chunk = {
                "id": f"chunk_detail_{chapter['id']}",
                "content": chapter["content"],
                "type": "text_detail",
                "chapter_id": chapter["id"],
                "metadata": {
                    "media_type": "text_detail",
                    "book_title": document_data["title"],
                    "source_text": chapter["content"][:200] + "...",
                    "context_link": {
                        "chapter_number": chapter["number"],
                        "chapter_title": chapter["title"],
                        "page_number": chapter["start_page"],
                    }
                }
            }
            chunks.append(detail_chunk)
        
        self.logger.info(f" {len(chunks)} ")
        return chunks
    
    async def _process_images(self, image_files: List[Path]) -> List[dict]:
        """"""
        
        self.logger.info(f" {len(image_files)} ")
        
        chunks = []
        
        for image_file in image_files:
            # TODO: 
            # 1. VLM
            # 2. 
            
            chunk = {
                "id": f"chunk_image_{int(time.time())}_{image_file.stem}",
                "content": f":  {image_file.stem} ",
                "type": "image", 
                "file_path": str(image_file),
                "metadata": {
                    "media_type": "image",
                    "book_title": "",
                    "source_text": f": {image_file.name}",
                    "source_url": str(image_file),
                    "context_link": {
                        "chapter_number": 1,
                        "chapter_title": "",
                        "page_number": 1,
                    }
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    async def _process_videos(self, video_files: List[Path]) -> List[dict]:
        """"""
        
        self.logger.info(f" {len(video_files)} ")
        
        chunks = []
        
        for video_file in video_files:
            # TODO: 
            # 1. 
            # 2. 
            # 3. 
            # 4. 
            
            chunk = {
                "id": f"chunk_video_{int(time.time())}_{video_file.stem}",
                "content": f":  {video_file.stem} ",
                "type": "video_clip",
                "file_path": str(video_file),
                "metadata": {
                    "media_type": "video_clip",
                    "book_title": "",
                    "source_text": f": {video_file.name}",
                    "source_url": str(video_file),
                    "context_link": {
                        "chapter_number": 1,
                        "chapter_title": "",
                        "page_number": 1,
                    }
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    async def _store_vectors(self, chunks: List[dict]) -> None:
        """"""
        
        self.logger.info(f" {len(chunks)} ")
        
        # TODO: 
        # 1. 
        # 2. 
        # 3. 
        
        for chunk in chunks:
            # 
            chunk["embedding"] = [0.1] * settings.embedding.dimension
        
        self.logger.info("")
    
    def _is_image_file(self, file_path: Path) -> bool:
        """"""
        extension = file_path.suffix.lower().lstrip('.')
        return extension in settings.media.image_formats_list
    
    def _is_video_file(self, file_path: Path) -> bool:
        """"""
        extension = file_path.suffix.lower().lstrip('.')
        return extension in settings.media.video_formats_list


async def main():
    """"""
    
    parser = argparse.ArgumentParser(description="RAG-SearX ")
    parser.add_argument("--book-path", required=True, help="")
    parser.add_argument("--media-path", help="")
    parser.add_argument("--book-title", help="")
    parser.add_argument("--author", help="")
    parser.add_argument("--language", default="zh", help="")
    
    args = parser.parse_args()
    
    logger.info("RAG-SearX ")
    logger.info(f": {args.book_path}")
    logger.info(f": {args.media_path}")
    
    builder = IndexBuilder()
    
    success = await builder.build_index(
        book_path=args.book_path,
        media_path=args.media_path,
        book_title=args.book_title,
        author=args.author,
        language=args.language
    )
    
    if success:
        logger.info("")
        sys.exit(0)
    else:
        logger.error("")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 