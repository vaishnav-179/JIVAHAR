import logging
import re
from typing import Tuple

from ai_module.gemma.gemma_service import GemmaService
from ai_module.prompts import format_notification

logger = logging.getLogger(__name__)

class NotificationGenerator:
    """
    Coordinates compiling donation details into action-oriented notifications
    for nearby volunteers and matching NGOs. Parses raw Gemma outputs using 
    regex patterns into distinct alert segments.
    """
    
    def __init__(self):
        self.gemma_service = GemmaService()

    def generate_notifications(
        self, 
        food_name: str, 
        quantity: str, 
        priority: str, 
        pickup_location: str
    ) -> Tuple[str, str]:
        """
        Generates and parses volunteer and NGO notifications from donation details.
        
        Args:
            food_name: Name of the food item.
            quantity: Amount of food.
            priority: Priority level (HIGH | MEDIUM | LOW).
            pickup_location: Address or general area for collection.
            
        Returns:
            A tuple: (volunteer_alert_string, ngo_alert_string)
        """
        if not food_name or not food_name.strip():
            raise ValueError("Food name cannot be empty.")
        if not quantity or not quantity.strip():
            raise ValueError("Quantity cannot be empty.")
        if not pickup_location or not pickup_location.strip():
            raise ValueError("Pickup location cannot be empty.")

        logger.info(f"Generating notifications for donation '{food_name}'...")
        
        # 1. Compile prompt using prompt templates
        system_instruction, user_prompt = format_notification(
            food_name=food_name,
            quantity=quantity,
            priority=priority,
            pickup_location=pickup_location
        )
        
        # 2. Run Gemma inference
        raw_response = self.gemma_service.generate_response(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.2  # Low temperature for highly precise output conforming to format
        )
        
        # 3. Parse raw response using regex patterns
        volunteer_alert = ""
        ngo_alert = ""
        
        # Regex search for Volunteer Notification
        vol_match = re.search(r"Volunteer Notification:\s*(.*)", raw_response, re.IGNORECASE)
        if vol_match:
            volunteer_alert = vol_match.group(1).strip()
        else:
            # Fallback parsing line by line
            for line in raw_response.split("\n"):
                if "volunteer" in line.lower() and ":" in line:
                    volunteer_alert = line.split(":", 1)[1].strip()
                    break
                    
        # Regex search for NGO Notification
        ngo_match = re.search(r"NGO Notification:\s*(.*)", raw_response, re.IGNORECASE)
        if ngo_match:
            ngo_alert = ngo_match.group(1).strip()
        else:
            # Fallback parsing line by line
            for line in raw_response.split("\n"):
                if "ngo" in line.lower() and ":" in line:
                    ngo_alert = line.split(":", 1)[1].strip()
                    break
                    
        # Clean any wrapping styling markers (asterisks, quotes) that LLM might introduce
        volunteer_alert = volunteer_alert.strip('*_"\' ')
        ngo_alert = ngo_alert.strip('*_"\' ')
        
        logger.info("Successfully parsed volunteer and NGO notification alerts.")
        return volunteer_alert, ngo_alert
