import unittest
from unittest.mock import MagicMock, patch
from ai_module.gemma import DonationSummaryGenerator

class TestDonationSummaryGenerator(unittest.TestCase):
    """
    Unit test suite validating the DonationSummaryGenerator component:
    verifying parameter mapping, prompt interpolation, and Gemma service integration.
    """
    
    @patch('ai_module.gemma.summary_generator.GemmaService')
    def test_generate_summary_flow(self, mock_gemma_class):
        # 1. Configure the GemmaService mock
        mock_gemma_instance = MagicMock()
        mock_gemma_class.return_value = mock_gemma_instance
        
        # Predefined mock structured summary
        mock_gemma_instance.generate_response.return_value = (
            "1. **Description**: 10 portions of fresh vegetable pasta cooked in tomato sauce.\n"
            "2. **Log Summary**: Fresh vegetable pasta (10 portions) listed for donation.\n"
            "3. **Logistics Recommendation**: Keep refrigerated. Pack in insulated containers."
        )
        
        # Initialize generator
        generator = DonationSummaryGenerator()
        
        # 2. Execute summary creation
        summary = generator.generate_summary(
            food_name="Vegetable Pasta",
            quantity="10 portions",
            prepared_time="1 hour ago",
            storage_condition="Refrigerated"
        )
        
        # 3. Assertions
        self.assertIn("Description", summary)
        self.assertIn("Log Summary", summary)
        self.assertIn("Logistics Recommendation", summary)
        
        # Verify correct prompt structures and variables are sent to GemmaService
        called_args, called_kwargs = mock_gemma_instance.generate_response.call_args
        
        # Check user prompt variables
        user_prompt = called_kwargs['prompt']
        self.assertIn("Food Item: Vegetable Pasta", user_prompt)
        self.assertIn("Quantity: 10 portions", user_prompt)
        self.assertIn("Prepared Time: 1 hour ago", user_prompt)
        self.assertIn("Storage Condition: Refrigerated", user_prompt)
        
        # Check system instructions
        system_instruction = called_kwargs['system_instruction']
        self.assertIn("food redistribution logistics", system_instruction.lower())
        self.assertIn("markdown format", system_instruction.lower())

if __name__ == "__main__":
    unittest.main()
