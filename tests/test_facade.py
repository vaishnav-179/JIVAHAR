import unittest
from unittest.mock import MagicMock, patch
from ai_module import JivaharAIFacade

class TestJivaharAIFacade(unittest.TestCase):
    """
    Unit test suite validating the JivaharAIFacade package facade:
    verifying method delegations, API parameters, and proper subclass routing.
    """
    
    @patch('ai_module.facade.FoodClassifier')
    @patch('ai_module.facade.DonationSummaryGenerator')
    @patch('ai_module.facade.FoodSafetyAdvisor')
    @patch('ai_module.facade.JivaharChatbot')
    @patch('ai_module.facade.RAGPipeline')
    @patch('ai_module.facade.NotificationGenerator')
    @patch('ai_module.facade.NGORecommendationExplainer')
    def test_facade_delegations(
        self,
        mock_match_explainer_class,
        mock_notification_class,
        mock_rag_pipeline_class,
        mock_chatbot_class,
        mock_safety_advisor_class,
        mock_summary_class,
        mock_classifier_class
    ):
        # Configure instances
        mock_classifier = mock_classifier_class.return_value
        mock_summary = mock_summary_class.return_value
        mock_safety_advisor = mock_safety_advisor_class.return_value
        mock_chatbot = mock_chatbot_class.return_value
        mock_rag_pipeline = mock_rag_pipeline_class.return_value
        mock_notification = mock_notification_class.return_value
        mock_match_explainer = mock_match_explainer_class.return_value
        
        # Instantiate Facade
        facade = JivaharAIFacade()
        
        # 1. Test get_safety_advice delegation
        facade.get_safety_advice("Veg Pasta", "1 hour ago", "Refrigerated")
        mock_safety_advisor.get_safety_advice.assert_called_once_with(
            food_name="Veg Pasta",
            prepared_time="1 hour ago",
            storage_condition="Refrigerated"
        )
        
        # 2. Test chat delegation
        facade.chat("Hi", [{"role": "user", "content": "Hi"}])
        mock_chatbot.chat.assert_called_once_with(
            user_message="Hi",
            chat_history=[{"role": "user", "content": "Hi"}]
        )
        
        # 3. Test generate_summary delegation
        facade.generate_summary("Veg Pasta", "10 portions", "1 hour ago", "Refrigerated")
        mock_summary.generate_summary.assert_called_once_with(
            food_name="Veg Pasta",
            quantity="10 portions",
            prepared_time="1 hour ago",
            storage_condition="Refrigerated"
        )
        
        # 4. Test generate_notifications delegation
        facade.generate_notifications("Veg Pasta", "10 portions", "HIGH", "Sector 12")
        mock_notification.generate_notifications.assert_called_once_with(
            food_name="Veg Pasta",
            quantity="10 portions",
            priority="HIGH",
            pickup_location="Sector 12"
        )
        
        # 5. Test explain_recommendation delegation
        facade.explain_recommendation("NGO", 2.5, 50.0, 4.8, "Veg Pasta")
        mock_match_explainer.explain_recommendation.assert_called_once_with(
            ngo_name="NGO",
            distance_km=2.5,
            capacity_kg=50.0,
            rating=4.8,
            food_details="Veg Pasta"
        )
        
        # 6. Test ingest_knowledge_base delegation
        facade.ingest_knowledge_base()
        mock_rag_pipeline.ingest_documents.assert_called_once()

    @patch('ai_module.integrated_pipeline.IntegratedDonationPipeline')
    @patch('ai_module.facade.FoodClassifier')
    @patch('ai_module.facade.DonationSummaryGenerator')
    @patch('ai_module.facade.FoodSafetyAdvisor')
    @patch('ai_module.facade.JivaharChatbot')
    @patch('ai_module.facade.RAGPipeline')
    @patch('ai_module.facade.NotificationGenerator')
    @patch('ai_module.facade.NGORecommendationExplainer')
    def test_facade_process_image_donation(
        self,
        mock_match_explainer_class,
        mock_notification_class,
        mock_rag_pipeline_class,
        mock_chatbot_class,
        mock_safety_advisor_class,
        mock_summary_class,
        mock_classifier_class,
        mock_integrated_pipeline_class
    ):
        mock_integrated_pipeline = mock_integrated_pipeline_class.return_value
        
        # Instantiate Facade
        facade = JivaharAIFacade()
        
        # Test process_image_donation delegation
        facade.process_image_donation("pic.jpg", "10 portions", "1 hour ago", "Refrigerated")
        mock_integrated_pipeline.process_image_donation.assert_called_once_with(
            image_path="pic.jpg",
            quantity="10 portions",
            prepared_time="1 hour ago",
            storage_condition="Refrigerated"
        )

if __name__ == "__main__":
    unittest.main()
