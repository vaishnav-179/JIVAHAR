import unittest
from unittest.mock import MagicMock, patch
from ai_module.gemma import NotificationGenerator

class TestNotificationGenerator(unittest.TestCase):
    """
    Unit test suite validating the NotificationGenerator component:
    verifying template formatting, regex output splitting, and alert length checks.
    """
    
    @patch('ai_module.gemma.notification_generator.GemmaService')
    def test_generate_notifications_flow(self, mock_gemma_class):
        # 1. Configure the GemmaService mock
        mock_gemma_instance = MagicMock()
        mock_gemma_class.return_value = mock_gemma_instance
        
        # Defined mock alerts
        mock_gemma_instance.generate_response.return_value = (
            "Volunteer Notification: Urgent pickup needed: 10 portions of Veg Pasta at Sector 12! Please accept now.\n"
            "NGO Notification: Match alert: 10 portions of Veg Pasta available at Sector 12. Claim within 15 mins."
        )
        
        # Initialize generator
        generator = NotificationGenerator()
        
        # 2. Execute notification generation
        vol_alert, ngo_alert = generator.generate_notifications(
            food_name="Veg Pasta",
            quantity="10 portions",
            priority="HIGH",
            pickup_location="Sector 12"
        )
        
        # 3. Assertions
        self.assertEqual(vol_alert, "Urgent pickup needed: 10 portions of Veg Pasta at Sector 12! Please accept now.")
        self.assertEqual(ngo_alert, "Match alert: 10 portions of Veg Pasta available at Sector 12. Claim within 15 mins.")
        
        # Check constraints (should be under 150 characters each)
        self.assertLess(len(vol_alert), 150, "Volunteer alert exceeds 150 character limit")
        self.assertLess(len(ngo_alert), 150, "NGO alert exceeds 150 character limit")
        
        # Verify correct parameters were sent to GemmaService
        called_args, called_kwargs = mock_gemma_instance.generate_response.call_args
        
        # Check user prompt variables
        user_prompt = called_kwargs['prompt']
        self.assertIn("Veg Pasta", user_prompt)
        self.assertIn("10 portions", user_prompt)
        self.assertIn("HIGH", user_prompt)
        self.assertIn("Sector 12", user_prompt)
        
        # Check system instructions
        system_instruction = called_kwargs['system_instruction']
        self.assertIn("notification generator", system_instruction.lower())
        self.assertIn("ultra-short", system_instruction.lower())

if __name__ == "__main__":
    unittest.main()
