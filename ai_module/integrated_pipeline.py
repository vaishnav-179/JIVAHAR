import logging
from typing import Dict, Any

from ai_module.cnn.food_classifier import FoodClassifier
from ai_module.gemma.summary_generator import DonationSummaryGenerator
from ai_module.rag.safety_advisor import FoodSafetyAdvisor

logger = logging.getLogger(__name__)

class IntegratedDonationPipeline:
    """
    Coordinates the end-to-end food donation pipeline:
    1. Classifies the food image using the pre-trained CNN to get the food category.
    2. Generates a structured donation summary using Gemma.
    3. Analyzes safety guidelines using FAISS index lookups and Gemma to retrieve safety advice.
    Returns a unified payload for the backend.
    """
    
    def __init__(self):
        logger.info("Initializing Integrated Donation Pipeline...")
        self.food_classifier = FoodClassifier()
        self.summary_generator = DonationSummaryGenerator()
        self.safety_advisor = FoodSafetyAdvisor()
        logger.info("Integrated Donation Pipeline components loaded successfully.")

    def process_image_donation(
        self, 
        image_path: str, 
        quantity: str, 
        prepared_time: str, 
        storage_condition: str
    ) -> Dict[str, Any]:
        """
        Processes an image-based food donation end-to-end.
        
        Args:
            image_path: Path to the image file of the food.
            quantity: Amount of food.
            prepared_time: Preparation time.
            storage_condition: Current storage temperature/condition.
            
        Returns:
            A dictionary containing:
              - food_name: The classified category name
              - cnn_confidence: Confidence score of the classification
              - summary: Gemma-generated log and description summary
              - safety_advice: Grounded safety report and volunteer checklist
              - safety_sources: List of cited DocumentChunk objects
        """
        logger.info(f"Processing image donation: {image_path}")
        
        # 1. Predict category using CNN
        food_name, cnn_confidence = self.food_classifier.predict(image_path)
        
        # 2. Compile structured summary
        summary = self.summary_generator.generate_summary(
            food_name=food_name,
            quantity=quantity,
            prepared_time=prepared_time,
            storage_condition=storage_condition
        )
        
        # 3. Retrieve safety rules and run safety advisor
        safety_advice, safety_sources = self.safety_advisor.get_safety_advice(
            food_name=food_name,
            prepared_time=prepared_time,
            storage_condition=storage_condition
        )
        
        # 4. Consolidate results
        payload = {
            "food_name": food_name,
            "cnn_confidence": cnn_confidence,
            "summary": summary,
            "safety_advice": safety_advice,
            "safety_sources": safety_sources
        }
        
        logger.info(f"Successfully processed image donation for category '{food_name}'.")
        return payload
