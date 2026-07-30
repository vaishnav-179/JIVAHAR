import os
import unittest
from pathlib import Path
from PIL import Image
from ai_module.cnn import FoodClassifier
from config.settings import settings

class TestFoodClassifier(unittest.TestCase):
    """
    Unit test suite validating the FoodClassifier component:
    verifying model initialization, state dict loading, image preprocessing, and inference output.
    """
    
    def setUp(self):
        # Create path for a temporary test image file
        self.temp_image_path = Path(__file__).resolve().parent / "temp_food_image.jpg"
        # Generate a dummy RGB image and save it to disk
        img = Image.new("RGB", (320, 320), color=(255, 0, 0))
        img.save(self.temp_image_path)

    def tearDown(self):
        # Remove the temporary test image
        if self.temp_image_path.exists():
            os.remove(self.temp_image_path)

    def test_food_classifier_inference(self):
        model_path = Path(settings.CNN_MODEL_PATH)
        if not model_path.exists():
            self.skipTest(f"Pre-trained weights file best_model.pth not found at: {model_path.resolve()}")
            
        # 1. Initialize classifier (loads best_model.pth)
        classifier = FoodClassifier()
        
        # 2. Verify model parameters loaded correctly
        self.assertIsNotNone(classifier.model, "Model not initialized")
        self.assertGreater(len(classifier.class_to_idx), 0, "Class mapping dictionary is empty")
        self.assertEqual(len(classifier.class_to_idx), 97, "Expected exactly 97 food classes mapping")
        
        # 3. Execute inference
        predicted_class, confidence = classifier.predict(str(self.temp_image_path))
        
        # 4. Assert prediction outputs mapping and value ranges
        self.assertIsInstance(predicted_class, str, "Predicted class should be a string")
        self.assertIn(predicted_class, classifier.class_to_idx, "Predicted class not found in mapping")
        self.assertIsInstance(confidence, float, "Confidence score should be a float")
        self.assertTrue(0.0 <= confidence <= 1.0, "Confidence score must be in bounds [0.0, 1.0]")
        
        print(f"\n[CNN TEST RESULT] Model predicted category: '{predicted_class}' with confidence: {confidence:.4f}")

if __name__ == "__main__":
    unittest.main()
