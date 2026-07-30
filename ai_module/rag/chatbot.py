import logging
from typing import List, Tuple, Optional

from ai_module.gemma.gemma_service import GemmaService
from ai_module.rag.document_processor import DocumentChunk
from ai_module.rag.vector_store import VectorStore
from ai_module.prompts import format_chatbot

logger = logging.getLogger(__name__)

class JivaharChatbot:
    """
    Handles natural language conversations with donors, volunteers, and NGOs.
    Queries the FAISS index to retrieve context-grounded platform FAQs and policies,
    formats conversation history to maintain dialogue state, and queries Gemma.
    """
    
    def __init__(self):
        self.gemma_service = GemmaService()
        self.vector_store = VectorStore()

    def _format_history(self, chat_history: Optional[List[dict]] = None) -> str:
        """
        Converts list-of-dict conversation history into a formatted chronological transcript.
        
        Args:
            chat_history: List of dicts, e.g. [{"role": "user", "content": "Hi"}, ...]
            
        Returns:
            A formatted string transcript.
        """
        if not chat_history:
            return ""
            
        formatted_messages = []
        for msg in chat_history:
            role = msg.get("role", "").lower()
            content = msg.get("content", "").strip()
            
            if not content:
                continue
                
            if role == "user":
                formatted_messages.append(f"User: {content}")
            elif role in ("assistant", "model", "bot"):
                formatted_messages.append(f"Jivahar Bot: {content}")
                
        return "\n".join(formatted_messages)

    def chat(
        self, 
        user_message: str, 
        chat_history: Optional[List[dict]] = None
    ) -> Tuple[str, List[DocumentChunk]]:
        """
        Performs a FAQ/policy similarity search, formats prompts with context 
        and dialogue history, and queries Gemma to generate a response.
        
        Args:
            user_message: The latest message from the user.
            chat_history: List of prior message dictionaries for context.
            
        Returns:
            A tuple: (chatbot_response_string, list_of_retrieved_source_chunks)
        """
        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        # 1. Semantic search FAISS index for relevant FAQ pages or policies
        # Querying with the user's latest query
        logger.info(f"Retrieving FAQ and policy context for chatbot query...")
        matches = self.vector_store.search(user_message, k=2)
        
        # 2. Extract context text blocks and citations
        context_blocks = []
        source_chunks = []
        for chunk, dist in matches:
            block = f"[Source: {chunk.source}, Page: {chunk.page}]\n{chunk.text}"
            context_blocks.append(block)
            source_chunks.append(chunk)
            
        context_str = "\n\n".join(context_blocks) if context_blocks else "No local guidelines found."
        
        # 3. Format history list into transcription block
        history_str = self._format_history(chat_history)
        
        # 4. Generate chatbot system instructions and dynamic prompts
        system_instruction, user_prompt = format_chatbot(
            user_message=user_message,
            chat_history=history_str,
            context=context_str
        )
        
        logger.info(f"Generating chatbot response from Gemma...")
        
        # 5. Run Gemma chatbot generation
        response = self.gemma_service.generate_response(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.3  # Slightly higher temperature for warmer, conversational tone
        )
        
        return response, source_chunks
