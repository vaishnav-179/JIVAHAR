import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """
    Data container representing a single text chunk extracted from a document.
    """
    text: str
    source: str
    page: int  # 1-indexed page number

class DocumentProcessor:
    """
    Handles PDF parsing, text extraction, and token/character-based chunking.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Safety guard for slicing step size
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be strictly greater than chunk_overlap ({chunk_overlap})."
            )

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Tuple[int, str]]:
        """
        Reads a PDF file and returns a list of tuples containing (page_number, text).
        
        Args:
            pdf_path: Path to the target PDF file.
            
        Returns:
            A list of tuples: (page_number_1_indexed, page_text_content)
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF document not found at: {pdf_path}")
            
        pages_content = []
        try:
            reader = PdfReader(str(pdf_path))
            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                text = page.extract_text()
                if text:
                    pages_content.append((page_num, text))
            logger.info(f"Successfully extracted {len(pages_content)} pages from {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to parse PDF {pdf_path}: {e}")
            raise e
            
        return pages_content

    def split_text_into_chunks(self, text: str, source_name: str, page_num: int) -> List[DocumentChunk]:
        """
        Splits a raw text string into smaller sliding-window character chunks.
        
        Args:
            text: Raw extracted text string.
            source_name: Name of the source file (for metadata).
            page_num: Page number from which text was extracted (for metadata).
            
        Returns:
            A list of DocumentChunk instances.
        """
        if not text or not text.strip():
            return []
            
        chunks = []
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text_len = len(text)
        
        start = 0
        step = self.chunk_size - self.chunk_overlap
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()
            
            # Save chunk if it contains actual content
            if chunk_text:
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    source=source_name,
                    page=page_num
                ))
                
            # Slide window forward
            start += step
            
        return chunks

    def process_directory(self, directory_path: Path) -> List[DocumentChunk]:
        """
        Iterates over all PDF documents in a folder, extracts text, chunks it,
        and aggregates all generated DocumentChunks.
        
        Args:
            directory_path: Folder containing PDF files.
            
        Returns:
            Combined list of DocumentChunk instances from all PDFs.
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        all_chunks = []
        pdf_files = list(directory_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF documents found in directory: {directory_path}")
            return []
            
        for pdf_path in pdf_files:
            pages = self.extract_text_from_pdf(pdf_path)
            for page_num, page_text in pages:
                chunks = self.split_text_into_chunks(page_text, pdf_path.name, page_num)
                all_chunks.extend(chunks)
                
        logger.info(f"Processed {len(pdf_files)} PDFs, generated {len(all_chunks)} chunks total.")
        return all_chunks
