import logging
from pathlib import Path
from typing import List, Tuple, Optional

from config.settings import settings
from ai_module.gemma.gemma_service import GemmaService
from ai_module.rag.document_processor import DocumentChunk, DocumentProcessor
from ai_module.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Standard RAG prompts to ensure response grounding and source citation
RAG_SYSTEM_INSTRUCTION = (
    "You are Jivahar RAG Advisor, an expert AI assistant for our food redistribution platform.\n"
    "Your goal is to answer the user's query using ONLY the provided document context.\n"
    "Always cite the source document name and page number for the facts you provide.\n"
    "If the context does not contain the answer, politely state that the information is not present in the knowledge base. "
    "Do not make up facts or use external knowledge."
)

RAG_USER_TEMPLATE = (
    "--- Reference Document Context ---\n"
    "{context}\n"
    "----------------------------------\n\n"
    "User Query: {query}\n"
    "Answer:"
)

class RAGPipeline:
    """
    Orchestrates the entire RAG pipeline: ingests PDF documents, searches the FAISS
    index, aggregates text context, compiles prompts, and runs Gemma inference.
    """
    
    def __init__(self):
        self.gemma_service = GemmaService()
        self.vector_store = VectorStore()

    def ingest_documents(self) -> None:
        """
        Ingests all PDF files in the knowledge base folder, chunks them,
        generates embeddings, and builds the FAISS index database.
        """
        kb_dir = Path(settings.BASE_DIR) / "data" / "knowledge_base"
        
        logger.info(f"Triggering knowledge base document ingestion from {kb_dir}...")
        
        # 1. Parse and chunk PDFs
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        chunks = processor.process_directory(kb_dir)
        
        if not chunks:
            raise RuntimeError(
                f"Ingestion failed: No chunks were processed from directory {kb_dir}. "
                "Ensure PDF documents exist in the data/knowledge_base folder."
            )
            
        # 2. Build and save index
        self.vector_store.build_index(chunks)
        logger.info("Ingestion completed successfully. Database index created and saved.")

    def query_with_context(
        self, 
        query_text: str, 
        k: int = 3, 
        temperature: float = 0.2
    ) -> Tuple[str, List[DocumentChunk]]:
        """
        Performs a semantic similarity search on FAISS, compiles matching context,
        and generates a grounded answer using Gemma.
        
        Args:
            query_text: The user search question.
            k: Number of matching document chunks to retrieve.
            temperature: Randomness of model generation.
            
        Returns:
            A tuple: (generated_answer_string, list_of_retrieved_source_chunks)
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty.")

        # 1. Retrieve closest matching chunks from FAISS index
        # search will automatically load the index from disk if not cached
        matches = self.vector_store.search(query_text, k=k)
        
        if not matches:
            logger.warning(f"No similarity matches found for query: '{query_text}'")
            return "No matching information was found in the knowledge base.", []

        # 2. Structure context entries with page and file citations
        context_blocks = []
        source_chunks = []
        
        for chunk, distance in matches:
            block = f"[Source File: {chunk.source}, Page: {chunk.page}]\n{chunk.text}"
            context_blocks.append(block)
            source_chunks.append(chunk)
            
        context_str = "\n\n".join(context_blocks)
        
        # 3. Format dynamic prompt
        user_prompt = RAG_USER_TEMPLATE.format(
            context=context_str,
            query=query_text
        )
        
        logger.info(f"Running Gemma inference for RAG query: '{query_text}'...")
        
        # 4. Generate grounded answer via Gemma
        answer = self.gemma_service.generate_response(
            prompt=user_prompt,
            system_instruction=RAG_SYSTEM_INSTRUCTION,
            temperature=temperature
        )
        
        return answer, source_chunks
