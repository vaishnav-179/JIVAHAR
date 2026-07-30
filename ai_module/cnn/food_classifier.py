import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from config.settings import settings

logger = logging.getLogger(__name__)

class FoodClassifier:
    """
    Loads the pre-trained EfficientNet-B3 food classification model (best_model.pth),
    preprocesses input images, and runs forward-pass inference to return predicted
    food categories and confidence scores.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else settings.CNN_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.idx_to_class: Dict[int, str] = {}
        self.class_to_idx: Dict[str, int] = {}
        
        # Standard preprocessing pipeline for EfficientNet-B3 (300x300 input resolution)
        self.preprocess = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self._load_model()

    def _load_model(self):
        """Loads model structure and weights from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pre-trained model weights not found at: {self.model_path.resolve()}. "
                "Ensure best_model.pth is present in the configured directory."
            )
            
        logger.info(f"Loading pre-trained CNN checkpoint from: {self.model_path.resolve()}...")
        checkpoint = torch.load(self.model_path, map_location="cpu")
        
        # Load class mappings
        self.class_to_idx = checkpoint.get("class_to_idx", {})
        if not self.class_to_idx:
            raise ValueError("Class mapping 'class_to_idx' not found in checkpoint.")
            
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        num_classes = len(self.class_to_idx)
        logger.info(f"Loaded class mapping for {num_classes} food categories.")
        
        # Initialize EfficientNet-B3 architecture
        logger.info("Initializing EfficientNet-B3 architecture...")
        self.model = models.efficientnet_b3(num_classes=num_classes)
        
        # Load weight state dict
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        logger.info("CNN model weights loaded successfully and set to evaluation mode.")

    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Runs image preprocessing and model inference.
        
        Args:
            image_path: Absolute path to the food image file.
            
        Returns:
            A tuple: (predicted_class_name, confidence_score)
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at: {img_path.resolve()}")
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open image file: {e}")
            
        # Preprocess and prepare batch dimension
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Inference (no gradient calculation)
        with torch.no_grad():
            outputs = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, index = torch.max(probabilities, dim=1)
            
        predicted_idx = index.item()
        confidence_score = confidence.item()
        
        predicted_class = self.idx_to_class.get(predicted_idx, "unknown")
        
        logger.info(f"Prediction: '{predicted_class}' with confidence {confidence_score:.4f}")
        return predicted_class, confidence_score
