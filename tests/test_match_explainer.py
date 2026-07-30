import unittest
from unittest.mock import MagicMock, patch
from ai_module.gemma import NGORecommendationExplainer

class TestNGORecommendationExplainer(unittest.TestCase):
    """
    Unit test suite validating the NGORecommendationExplainer component:
    verifying prompt compilation templates, input validation checks, and Gemma integration.
    """
    
    @patch('ai_module.gemma.match_explainer.GemmaService')
    def test_explain_recommendation_flow(self, mock_gemma_class):
        # 1. Configure the GemmaService mock
        mock_gemma_instance = MagicMock()
        mock_gemma_class.return_value = mock_gemma_instance
        
        mock_gemma_instance.generate_response.return_value = (
            "Helping Hands NGO is the optimal match for this donation because they are located only 2.5 km away, "
            "minimizing transit time for the Veg Pasta. Furthermore, they have 50.0 kg of capacity remaining and "
            "hold a high 4.8 star rating, ensuring reliable handling and immediate distribution."
        )
        
        # Initialize explainer
        explainer = NGORecommendationExplainer()
        
        # 2. Execute matching explanation
        explanation = explainer.explain_recommendation(
            ngo_name="Helping Hands NGO",
            distance_km=2.5,
            capacity_kg=50.0,
            rating=4.8,
            food_details="Veg Pasta"
        )
        
        # 3. Assertions
        self.assertIn("Helping Hands NGO", explanation)
        self.assertIn("2.5 km", explanation)
        self.assertIn("50.0 kg", explanation)
        self.assertIn("4.8 star", explanation)
        
        # Verify correct parameters were sent to GemmaService
        called_args, called_kwargs = mock_gemma_instance.generate_response.call_args
        
        # Check user prompt variables
        user_prompt = called_kwargs['prompt']
        self.assertIn("NGO Name: Helping Hands NGO", user_prompt)
        self.assertIn("Distance to Donor: 2.5 km", user_prompt)
        self.assertIn("Capacity Available: 50.0 kg", user_prompt)
        self.assertIn("NGO Rating: 4.8/5", user_prompt)
        self.assertIn("Food Donation Details: Veg Pasta", user_prompt)
        
        # Check system instructions
        system_instruction = called_kwargs['system_instruction']
        self.assertIn("matching advisor", system_instruction.lower())
        self.assertIn("justification", system_instruction.lower())

    def test_validation_checks(self):
        # Verify input validation checks raise expected exceptions
        explainer = NGORecommendationExplainer()
        
        # Invalid NGO Name (empty)
        with self.assertRaises(ValueError):
            explainer.explain_recommendation("", 2.5, 50.0, 4.8, "Veg Pasta")
            
        # Negative Distance
        with self.assertRaises(ValueError):
            explainer.explain_recommendation("NGO", -1.0, 50.0, 4.8, "Veg Pasta")
            
        # Negative Capacity
        with self.assertRaises(ValueError):
            explainer.explain_recommendation("NGO", 2.5, -10.0, 4.8, "Veg Pasta")
            
        # Invalid Rating (above 5.0)
        with self.assertRaises(ValueError):
            explainer.explain_recommendation("NGO", 2.5, 50.0, 6.0, "Veg Pasta")
            
        # Invalid Food Details (empty)
        with self.assertRaises(ValueError):
            explainer.explain_recommendation("NGO", 2.5, 50.0, 4.8, "")

if __name__ == "__main__":
    unittest.main()
