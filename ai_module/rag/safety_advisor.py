import logging
from typing import List, Tuple

from ai_module.gemma.gemma_service import GemmaService
from ai_module.rag.document_processor import DocumentChunk
from ai_module.rag.vector_store import VectorStore
from ai_module.prompts import format_food_safety

logger = logging.getLogger(__name__)

class FoodSafetyAdvisor:
    """
    Coordinates food safety query parsing, retrieves regulatory and hygiene codes 
    from the FAISS index, compiles the custom safety prompts, and queries Gemma
    to return detailed safety assessments and pickup priorities.
    """
    
    def __init__(self):
        self.gemma_service = GemmaService()
        self.vector_store = VectorStore()

    def get_safety_advice(
        self, 
        food_name: str, 
        prepared_time: str, 
        storage_condition: str
    ) -> Tuple[str, List[DocumentChunk]]:
        """
        Runs document search, formats safety templates, and returns natural language
        safety advice with priority scores and volunteer checklists.
        
        Args:
            food_name: Name of the food item.
            prepared_time: Timestamp or time elapsed since preparation.
            storage_condition: Storage temperature or method (e.g. Refrigerated, Room Temp).
            
        Returns:
            A tuple: (safety_advice_markdown_string, list_of_retrieved_source_chunks)
        """
        if not food_name or not food_name.strip():
            raise ValueError("Food name cannot be empty.")
            
        # 1. Compile semantic search query focused on safety rules for this food type
        search_query = f"food safety rules, temperature danger zone guidelines, storage shelf life for {food_name}"
        
        logger.info(f"Retrieving safety guidelines for '{food_name}'...")
        
        # 2. Query FAISS index for relevant safety manual segments
        matches = self.vector_store.search(search_query, k=2)
        
        # 3. Format retrieved context block
        context_blocks = []
        source_chunks = []
        for chunk, dist in matches:
            block = f"[Source: {chunk.source}, Page: {chunk.page}]\n{chunk.text}"
            context_blocks.append(block)
            source_chunks.append(chunk)
            
        context_str = "\n\n".join(context_blocks) if context_blocks else "No local safety guidelines found."
        
        # 4. Generate system and user prompts using Phase 4 prompts module
        system_instruction, user_prompt = format_food_safety(
            food_name=food_name,
            prepared_time=prepared_time,
            storage_condition=storage_condition,
            context=context_str
        )
        
        logger.info(f"Generating safety assessment from Gemma for '{food_name}'...")
        
        # 5. Execute Gemma safety reasoning
        advice = self.gemma_service.generate_response(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.1  # Low temperature for highly deterministic, safe answers
        )
        
        return advice, source_chunks
