#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow

"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import pypdf as PyPDF2
import pdfplumber


class ChunkingStrategy(Enum):
    """"""
    NAIVE = "naive"  # 
    BOOK = "book"    # 
    MANUAL = "manual"  # 
    QA = "qa"        # 
    TABLE = "table"  # 
    PAPER = "paper"  # 
    PRESENTATION = "presentation"  # 
    LAWS = "laws"    # 
    EMAIL = "email"  # 
    ONE = "one"      # 


@dataclass
class ChunkConfig:
    """"""
    strategy: ChunkingStrategy = ChunkingStrategy.BOOK
    chunk_token_count: int = 128   # chunktoken
    chunk_overlap: float = 0.2     # 
    auto_keywords: bool = True     # 
    auto_question: bool = True     # 


@dataclass
class DocumentChunk:
    """"""
    chunk_id: str
    content: str
    page_number: Optional[int] = None
    chapter_title: Optional[str] = None
    keywords: List[str] = None
    questions: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.questions is None:
            self.questions = []
        if self.metadata is None:
            self.metadata = {}


class RAGFlowDocumentProcessor:
    """RAGFlow"""
    
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        self.chinese_punctuation = r'[""''…—―‐–]'
        self.english_punctuation = r'[.!?;,:\'"()\[\]{}<>…\-–—]'
        
    def process_document(self, file_path: str, **kwargs) -> List[DocumentChunk]:
        """"""
        
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        # 
        if suffix == '.pdf':
            return self._process_pdf(file_path, **kwargs)
        elif suffix == '.txt':
            return self._process_text(file_path, **kwargs)
        else:
            raise ValueError(f": {suffix}")
    
    def _process_pdf(self, file_path: Path, **kwargs) -> List[DocumentChunk]:
        """PDF"""
        
        # PDF
        pdf_data = self._extract_pdf_content(file_path)
        if not pdf_data:
            return []
        
        all_chunks = []
        
        for page_data in pdf_data["pages"]:
            page_num = page_data["page_number"]
            page_text = page_data["text"]
            
            if not page_text.strip():
                continue
            
            # 
            page_chunks = self._chunk_text_by_strategy(
                text=page_text,
                page_number=page_num,
                strategy=self.config.strategy
            )
            
            all_chunks.extend(page_chunks)
        
        # 
        if self.config.auto_keywords or self.config.auto_question:
            all_chunks = self._enhance_chunks(all_chunks)
        
        return all_chunks
    
    def _process_text(self, file_path: Path, **kwargs) -> List[DocumentChunk]:
        """"""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f": {e}")
        
        # 
        chunks = self._chunk_text_by_strategy(
            text=content,
            strategy=self.config.strategy
        )
        
        # 
        if self.config.auto_keywords or self.config.auto_question:
            chunks = self._enhance_chunks(chunks)
        
        return chunks
    
    def _extract_pdf_content(self, file_path: Path) -> Optional[Dict]:
        """PDF"""
        
        # pdfplumber
        try:
            pages_data = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_data.append({
                            "page_number": page_num,
                            "text": text.strip()
                        })
            
            if pages_data:
                return {
                    "method": "pdfplumber",
                    "total_pages": len(pages_data),
                    "pages": pages_data
                }
        except Exception as e:
            print(f"pdfplumber: {e}")
        
        # PyPDF2
        try:
            pages_data = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_data.append({
                            "page_number": page_num,
                            "text": text.strip()
                        })
            
            if pages_data:
                return {
                    "method": "PyPDF2",
                    "total_pages": len(pages_data),
                    "pages": pages_data
                }
        except Exception as e:
            print(f"PyPDF2: {e}")
        
        return None
    
    def _chunk_text_by_strategy(
        self, 
        text: str, 
        page_number: Optional[int] = None,
        strategy: ChunkingStrategy = None
    ) -> List[DocumentChunk]:
        """"""
        
        strategy = strategy or self.config.strategy
        
        if strategy == ChunkingStrategy.BOOK:
            return self._chunk_book_strategy(text, page_number)
        elif strategy == ChunkingStrategy.PAPER:
            return self._chunk_paper_strategy(text, page_number)
        elif strategy == ChunkingStrategy.QA:
            return self._chunk_qa_strategy(text, page_number)
        elif strategy == ChunkingStrategy.TABLE:
            return self._chunk_table_strategy(text, page_number)
        else:
            # 
            return self._chunk_book_strategy(text, page_number)
    
    def _chunk_book_strategy(self, text: str, page_number: Optional[int] = None) -> List[DocumentChunk]:
        """ - RAGFlow"""
        
        chunks = []
        
        # 
        chapter_pattern = r'[\d]+[]\s*[:]\s*(.+?)(?=\n|$)'
        chapters = re.split(chapter_pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        
        if len(chapters) > 1:
            # 
            for i in range(1, len(chapters), 2):
                if i + 1 < len(chapters):
                    chapter_title = chapters[i].strip()
                    chapter_content = chapters[i + 1].strip()
                    
                    if chapter_content:
                        chapter_chunks = self._split_text_semantic(
                            chapter_content, 
                            max_tokens=self.config.chunk_token_count,
                            overlap_ratio=self.config.chunk_overlap
                        )
                        
                        for j, chunk_text in enumerate(chapter_chunks):
                            chunk_id = self._generate_chunk_id(chunk_text, page_number, j)
                            chunks.append(DocumentChunk(
                                chunk_id=chunk_id,
                                content=chunk_text,
                                page_number=page_number,
                                chapter_title=chapter_title,
                                metadata={"chunk_index": j, "total_chunks": len(chapter_chunks)}
                            ))
        else:
            # 
            text_chunks = self._split_text_semantic(
                text,
                max_tokens=self.config.chunk_token_count,
                overlap_ratio=self.config.chunk_overlap
            )
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_id = self._generate_chunk_id(chunk_text, page_number, i)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    page_number=page_number,
                    metadata={"chunk_index": i, "total_chunks": len(text_chunks)}
                ))
        
        return chunks
    
    def _chunk_paper_strategy(self, text: str, page_number: Optional[int] = None) -> List[DocumentChunk]:
        """"""
        
        chunks = []
        
        # Abstract, Introduction, Methodology, Results, Conclusion
        section_patterns = [
            r'(?:Abstract|)\s*[:]?\s*',
            r'(?:Introduction||)\s*[:]?\s*',
            r'(?:Methodology||)\s*[:]?\s*',
            r'(?:Results|)\s*[:]?\s*',
            r'(?:Discussion|)\s*[:]?\s*',
            r'(?:Conclusion|)\s*[:]?\s*',
            r'(?:References|)\s*[:]?\s*'
        ]
        
        # 
        current_section = "Unknown"
        sections = []
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
            for match in matches:
                sections.append((match.start(), match.group().strip()))
        
        sections.sort(key=lambda x: x[0])  # 
        
        if sections:
            for i, (start_pos, section_name) in enumerate(sections):
                end_pos = sections[i + 1][0] if i + 1 < len(sections) else len(text)
                section_content = text[start_pos:end_pos].strip()
                
                if section_content:
                    section_chunks = self._split_text_semantic(
                        section_content,
                        max_tokens=self.config.chunk_token_count,
                        overlap_ratio=self.config.chunk_overlap
                    )
                    
                    for j, chunk_text in enumerate(section_chunks):
                        chunk_id = self._generate_chunk_id(chunk_text, page_number, j)
                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            content=chunk_text,
                            page_number=page_number,
                            chapter_title=section_name,
                            metadata={"section": section_name, "chunk_index": j}
                        ))
        else:
            # 
            chunks = self._chunk_book_strategy(text, page_number)
        
        return chunks
    
    def _chunk_qa_strategy(self, text: str, page_number: Optional[int] = None) -> List[DocumentChunk]:
        """"""
        
        chunks = []
        
        # 
        qa_patterns = [
            r'[]?\s*[:]\s*(.+?)\s*[]?\s*[:]\s*(.+?)(?=[]?|$)',
            r'Q\s*[:]\s*(.+?)\s*A\s*[:]\s*(.+?)(?=Q|$)',
            r'(\d+)\.\s*(.+?)\s*[]?\s*[:]\s*(.+?)(?=\d+\.|$)'
        ]
        
        found_qa = False
        for pattern in qa_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            
            if matches:
                found_qa = True
                for i, match in enumerate(matches):
                    if len(match) >= 2:
                        question = match[-2].strip()  # 
                        answer = match[-1].strip()    # 
                        
                        qa_text = f": {question}\n\n: {answer}"
                        chunk_id = self._generate_chunk_id(qa_text, page_number, i)
                        
                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            content=qa_text,
                            page_number=page_number,
                            questions=[question],
                            metadata={"type": "qa_pair", "question": question, "answer": answer}
                        ))
                break
        
        if not found_qa:
            # 
            chunks = self._chunk_book_strategy(text, page_number)
        
        return chunks
    
    def _chunk_table_strategy(self, text: str, page_number: Optional[int] = None) -> List[DocumentChunk]:
        """"""
        
        chunks = []
        
        # |
        lines = text.split('\n')
        table_lines = []
        current_table = []
        
        for line in lines:
            line = line.strip()
            if '|' in line and len(line.split('|')) >= 3:  # 3
                current_table.append(line)
            else:
                if current_table:
                    table_lines.append('\n'.join(current_table))
                    current_table = []
                
                # 
                if line:
                    text_chunks = self._split_text_semantic(
                        line,
                        max_tokens=self.config.chunk_token_count,
                        overlap_ratio=0  # 
                    )
                    
                    for i, chunk_text in enumerate(text_chunks):
                        chunk_id = self._generate_chunk_id(chunk_text, page_number, len(chunks))
                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            content=chunk_text,
                            page_number=page_number,
                            metadata={"type": "text"}
                        ))
        
        # 
        if current_table:
            table_lines.append('\n'.join(current_table))
        
        # chunk
        for i, table_text in enumerate(table_lines):
            chunk_id = self._generate_chunk_id(table_text, page_number, len(chunks))
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=table_text,
                page_number=page_number,
                metadata={"type": "table", "table_index": i}
            ))
        
        return chunks
    
    def _split_text_semantic(self, text: str, max_tokens: int = 128, overlap_ratio: float = 0.2) -> List[str]:
        """ - RAGFlow"""
        
        chunks = []
        
        # 
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            # token≈1token≈1token
            paragraph_tokens = self._estimate_tokens(paragraph)
            current_tokens = self._estimate_tokens(current_chunk)
            
            if current_tokens + paragraph_tokens <= max_tokens:
                # chunk
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                # chunkchunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # max_tokens
                if paragraph_tokens > max_tokens:
                    para_chunks = self._split_paragraph(paragraph, max_tokens)
                    chunks.extend(para_chunks[:-1])  # 
                    current_chunk = para_chunks[-1] if para_chunks else ""
                else:
                    current_chunk = paragraph
        
        # chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # 
        if overlap_ratio > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks, overlap_ratio)
        
        return chunks
    
    def _split_paragraph(self, paragraph: str, max_tokens: int) -> List[str]:
        """"""
        
        chunks = []
        
        # 
        sentences = re.split(r'[.!?]', paragraph)
        sentences = [s.strip() + '' for s in sentences if s.strip()]
        
        current_chunk = ""
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            current_tokens = self._estimate_tokens(current_chunk)
            
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # max_tokens
                if sentence_tokens > max_tokens:
                    # 
                    char_limit = max_tokens * 2  # 
                    for i in range(0, len(sentence), char_limit):
                        chunks.append(sentence[i:i + char_limit])
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str], overlap_ratio: float) -> List[str]:
        """"""
        
        if not chunks or overlap_ratio <= 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # chunk
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # 
            overlap_chars = int(len(prev_chunk) * overlap_ratio)
            
            if overlap_chars > 0:
                # chunk
                overlap_text = prev_chunk[-overlap_chars:]
                overlapped_chunk = overlap_text + "\n\n" + current_chunk
                overlapped_chunks.append(overlapped_chunk)
            else:
                overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """token"""
        
        if not text:
            return 0
        
        # ≈1token≈1token≈0.5token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        punctuation = len(re.findall(r'[^\u4e00-\u9fff\w\s]', text))
        
        estimated_tokens = chinese_chars + english_words + (punctuation * 0.5)
        return int(estimated_tokens)
    
    def _generate_chunk_id(self, content: str, page_number: Optional[int] = None, index: int = 0) -> str:
        """chunk ID"""
        
        base_string = f"{content}_{page_number}_{index}"
        return hashlib.md5(base_string.encode('utf-8')).hexdigest()
    
    def _enhance_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """chunks - """
        
        enhanced_chunks = []
        
        for chunk in chunks:
            enhanced_chunk = chunk
            
            # 
            if self.config.auto_keywords:
                enhanced_chunk.keywords = self._extract_keywords(chunk.content)
            
            # 
            if self.config.auto_question:
                enhanced_chunk.questions = self._generate_questions(chunk.content)
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _extract_keywords(self, text: str) -> List[str]:
        """"""
        
        # 
        clean_text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
        
        # 
        words = []
        
        # 2-4
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', clean_text)
        words.extend(chinese_words)
        
        # 
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_text)
        words.extend(english_words)
        
        # 
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 5
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word for word, freq in keywords if freq > 1]  # 2
    
    def _generate_questions(self, text: str) -> List[str]:
        """"""
        
        questions = []
        
        # 
        if '' not in text and '' not in text:
            # ""
            concepts = self._extract_keywords(text)
            if concepts:
                questions.append(f"{concepts[0]}")
        
        if '' not in text and '' not in text:
            # ""
            verbs = re.findall(r'[\u4e00-\u9fff]*[||||][\u4e00-\u9fff]*', text)
            if verbs:
                questions.append(f"{verbs[0]}")
        
        if '' not in text:
            # ""
            if '' in text or '' in text:
                questions.append("")
        
        return questions[:3]  # 3


# 
if __name__ == "__main__":
    # 
    config = ChunkConfig(
        strategy=ChunkingStrategy.BOOK,
        chunk_token_count=128,
        chunk_overlap=0.2,
        auto_keywords=True,
        auto_question=True
    )
    
    # 
    processor = RAGFlowDocumentProcessor(config)
    
    # 
    try:
        chunks = processor.process_document("ragtest.pdf")
        print(f" {len(chunks)} ")
        
        for i, chunk in enumerate(chunks[:3]):  # 3
            print(f"\n--- Chunk {i+1} ---")
            print(f"ID: {chunk.chunk_id[:16]}...")
            print(f"Page: {chunk.page_number}")
            print(f"Chapter: {chunk.chapter_title}")
            print(f"Content: {chunk.content[:200]}...")
            print(f"Keywords: {chunk.keywords}")
            print(f"Questions: {chunk.questions}")
            
    except Exception as e:
        print(f": {e}") 