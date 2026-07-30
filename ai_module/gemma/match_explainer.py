import logging
from ai_module.gemma.gemma_service import GemmaService
from ai_module.prompts import format_recommendation_explanation

logger = logging.getLogger(__name__)

class NGORecommendationExplainer:
    """
    Coordinates compiling NGO match decisions and metrics (distance, capacity, rating)
    into standard narrative paragraphs justifying the selection.
    """
    
    def __init__(self):
        self.gemma_service = GemmaService()

    def explain_recommendation(
        self, 
        ngo_name: str, 
        distance_km: float, 
        capacity_kg: float, 
        rating: float, 
        food_details: str
    ) -> str:
        """
        Generates a professional justification paragraph explaining why the NGO was matched.
        
        Args:
            ngo_name: Name of the matched NGO.
            distance_km: Distance in kilometers from donor to NGO.
            capacity_kg: Storage capacity in kilograms available at the NGO.
            rating: Average feedback rating of the NGO (out of 5).
            food_details: Brief description of the food donation.
            
        Returns:
            A single paragraph justification string.
        """
        if not ngo_name or not ngo_name.strip():
            raise ValueError("NGO name cannot be empty.")
        if distance_km < 0:
            raise ValueError("Distance cannot be negative.")
        if capacity_kg < 0:
            raise ValueError("Capacity cannot be negative.")
        if not 0 <= rating <= 5:
            raise ValueError("Rating must be between 0 and 5.")
        if not food_details or not food_details.strip():
            raise ValueError("Food details cannot be empty.")

        logger.info(f"Generating matching justification explanation for '{ngo_name}'...")
        
        # 1. Compile prompt using prompt templates
        system_instruction, user_prompt = format_recommendation_explanation(
            ngo_name=ngo_name,
            distance_km=distance_km,
            capacity_kg=capacity_kg,
            rating=rating,
            food_details=food_details
        )
        
        # 2. Run Gemma inference
        explanation = self.gemma_service.generate_response(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.2  # Low temperature for highly precise, non-embellished justifications
        )
        
        return explanation.strip()
