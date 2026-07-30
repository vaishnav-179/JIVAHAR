import logging
from ai_module.gemma.gemma_service import GemmaService
from ai_module.prompts import format_donation_summary

logger = logging.getLogger(__name__)

class DonationSummaryGenerator:
    """
    Coordinates compiling donation details into structured logs, description summaries,
    and immediate logistics advice using Gemma inference.
    """
    
    def __init__(self):
        self.gemma_service = GemmaService()

    def generate_summary(
        self, 
        food_name: str, 
        quantity: str, 
        prepared_time: str, 
        storage_condition: str
    ) -> str:
        """
        Synthesizes donation details and returns a standardized structured summary report.
        
        Args:
            food_name: Name of the food item.
            quantity: Amount of food (e.g. 5 kg, 20 portions).
            prepared_time: When the food was prepared.
            storage_condition: How the food is stored.
            
        Returns:
            A structured markdown summary containing description, log summary,
            and logistics advice.
        """
        if not food_name or not food_name.strip():
            raise ValueError("Food name cannot be empty.")
        if not quantity or not quantity.strip():
            raise ValueError("Quantity cannot be empty.")

        logger.info(f"Generating structured donation summary for '{food_name}'...")
        
        # 1. Format prompts using prompt engineering templates
        system_instruction, user_prompt = format_donation_summary(
            food_name=food_name,
            quantity=quantity,
            prepared_time=prepared_time,
            storage_condition=storage_condition
        )
        
        # 2. Run Gemma inference
        summary = self.gemma_service.generate_response(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.1  # Low temperature for highly structured, precise outputs
        )
        
        return summary
