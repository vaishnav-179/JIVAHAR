import unittest
from ai_module.prompts import (
    format_donation_summary,
    format_food_safety,
    format_chatbot,
    format_notification,
    format_recommendation_explanation
)

class TestPrompts(unittest.TestCase):
    """
    Validates that the string interpolation and formatting logic
    within the prompt engineering module works correctly for all 5 tasks.
    """
    
    def test_format_donation_summary(self):
        sys, user = format_donation_summary(
            food_name="Vegetable Biryani",
            quantity="10 portions",
            prepared_time="2 hours ago",
            storage_condition="Refrigerated"
        )
        # Check system prompt exists and user prompt interpolates variables correctly
        self.assertTrue(len(sys) > 0)
        self.assertIn("Vegetable Biryani", user)
        self.assertIn("10 portions", user)
        self.assertIn("2 hours ago", user)
        self.assertIn("Refrigerated", user)

    def test_format_food_safety_with_context(self):
        sys, user = format_food_safety(
            food_name="Egg Salad",
            prepared_time="4 hours ago",
            storage_condition="Room Temperature",
            context="Rule: Perishables at room temp over 2 hours must be flagged."
        )
        self.assertTrue(len(sys) > 0)
        self.assertIn("Egg Salad", user)
        self.assertIn("4 hours ago", user)
        self.assertIn("Room Temperature", user)
        self.assertIn("Rule: Perishables at room temp", user)

    def test_format_food_safety_default_context(self):
        # Test default fallback when no context is provided
        _, user = format_food_safety(
            food_name="Canned Beans",
            prepared_time="1 week ago",
            storage_condition="Pantry"
        )
        self.assertIn("Canned Beans", user)
        self.assertIn("No additional policy context available", user)

    def test_format_chatbot(self):
        sys, user = format_chatbot(
            user_message="Can I volunteer?",
            chat_history="User: Hello\nJivahar Bot: Welcome!",
            context="FAQ: Volunteers sign up via the register page."
        )
        self.assertTrue(len(sys) > 0)
        self.assertIn("Can I volunteer?", user)
        self.assertIn("Welcome!", user)
        self.assertIn("FAQ: Volunteers sign up", user)

    def test_format_notification(self):
        sys, user = format_notification(
            food_name="Fresh Fruit Cups",
            quantity="50 units",
            priority="HIGH",
            pickup_location="Community Center Kitchen"
        )
        self.assertTrue(len(sys) > 0)
        self.assertIn("Fresh Fruit Cups", user)
        self.assertIn("50 units", user)
        self.assertIn("HIGH", user)
        self.assertIn("Community Center Kitchen", user)

    def test_format_recommendation_explanation(self):
        sys, user = format_recommendation_explanation(
            ngo_name="Safe Haven Shelter",
            distance_km=4.2,
            capacity_kg=120.0,
            rating=4.9,
            food_details="Warm Lentil Soup"
        )
        self.assertTrue(len(sys) > 0)
        self.assertIn("Safe Haven Shelter", user)
        self.assertIn("4.2", user)
        self.assertIn("120", user)
        self.assertIn("4.9", user)
        self.assertIn("Warm Lentil Soup", user)

if __name__ == "__main__":
    unittest.main()
