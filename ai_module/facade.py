import logging
from typing import Dict, Any, List, Tuple, Optional

from ai_module.cnn.food_classifier import FoodClassifier
from ai_module.gemma.summary_generator import DonationSummaryGenerator
from ai_module.gemma.notification_generator import NotificationGenerator
from ai_module.gemma.match_explainer import NGORecommendationExplainer
from ai_module.rag.safety_advisor import FoodSafetyAdvisor
from ai_module.rag.chatbot import JivaharChatbot
from ai_module.rag.rag_pipeline import RAGPipeline
from ai_module.rag.document_processor import DocumentChunk

logger = logging.getLogger(__name__)

class JivaharAIFacade:
    """
    Unified Facade wrapper class for the Jivahar AI/ML module.
    Exposes clean, packaged method interfaces for image classification, RAG Q&A,
    donation summary logging, alert notifications, chatbot dialogue, and matching justifications.
    """
    
    def __init__(self):
        logger.info("Packaging Jivahar AI Facade API...")
        # 1. Image Classification & Integrated Pipeline components
        self.food_classifier = FoodClassifier()
        self.summary_generator = DonationSummaryGenerator()
        
        # 2. RAG & Safety Guidelines components
        self.safety_advisor = FoodSafetyAdvisor()
        self.chatbot = JivaharChatbot()
        self.rag_pipeline = RAGPipeline()
        
        # 3. Message Notification & Explanatory AI components
        self.notification_generator = NotificationGenerator()
        self.match_explainer = NGORecommendationExplainer()
        logger.info("Jivahar AI Facade successfully initialized.")

    def process_image_donation(
        self, 
        image_path: str, 
        quantity: str, 
        prepared_time: str, 
        storage_condition: str
    ) -> Dict[str, Any]:
        """
        Runs the end-to-end image-based donation pipeline.
        Predicts category name from photo, generates log summaries, and retrieves safety advisor reports.
        """
        # Load integrated donation pipeline on-demand or delegate to cached component pipeline
        from ai_module.integrated_pipeline import IntegratedDonationPipeline
        pipeline = IntegratedDonationPipeline()
        return pipeline.process_image_donation(
            image_path=image_path,
            quantity=quantity,
            prepared_time=prepared_time,
            storage_condition=storage_condition
        )

    def get_safety_advice(
        self, 
        food_name: str, 
        prepared_time: str, 
        storage_condition: str
    ) -> Tuple[str, List[DocumentChunk]]:
        """
        Queries food safety regulations to generate consumption advice and inspection checklists.
        """
        return self.safety_advisor.get_safety_advice(
            food_name=food_name,
            prepared_time=prepared_time,
            storage_condition=storage_condition
        )

    def chat(
        self, 
        user_message: str, 
        chat_history: Optional[List[dict]] = None
    ) -> Tuple[str, List[DocumentChunk]]:
        """
        Answers general user queries regarding platform logistics, rules, and guidelines.
        """
        return self.chatbot.chat(
            user_message=user_message,
            chat_history=chat_history
        )

    def generate_summary(
        self, 
        food_name: str, 
        quantity: str, 
        prepared_time: str, 
        storage_condition: str
    ) -> str:
        """
        Generates standard structured text descriptions and logistics logs for donation records.
        """
        return self.summary_generator.generate_summary(
            food_name=food_name,
            quantity=quantity,
            prepared_time=prepared_time,
            storage_condition=storage_condition
        )

    def generate_notifications(
        self, 
        food_name: str, 
        quantity: str, 
        priority: str, 
        pickup_location: str
    ) -> Tuple[str, str]:
        """
        Generates short notification alerts (under 150 characters) for volunteers and matching NGOs.
        """
        return self.notification_generator.generate_notifications(
            food_name=food_name,
            quantity=quantity,
            priority=priority,
            pickup_location=pickup_location
        )

    def explain_recommendation(
        self, 
        ngo_name: str, 
        distance_km: float, 
        capacity_kg: float, 
        rating: float, 
        food_details: str,
        context: Optional[str] = None
    ) -> str:
        """
        Writes a justification paragraph explaining why the NGO was matched to this donation.
        """
        return self.match_explainer.explain_recommendation(
            ngo_name=ngo_name,
            distance_km=distance_km,
            capacity_kg=capacity_kg,
            rating=rating,
            food_details=food_details,
            context=context
        )

    def ingest_knowledge_base(self):
        """
        Ingests source PDF documents to rebuild the local FAISS semantic index database.
        Usually executed by platform administrators when manuals or policy documents are updated.
        """
        logger.info("Triggering knowledge base index re-ingestion...")
        self.rag_pipeline.ingest_documents()
        logger.info("Knowledge base index updated successfully.")
