# ==============================================================================
# NEONEXUS AI MODULE
# This package exposes production-ready API interfaces for Gemma integration,
# RAG querying, and CNN food classification to the Flask backend application.
# ==============================================================================

from ai_module.integrated_pipeline import IntegratedDonationPipeline
from ai_module.facade import JivaharAIFacade

__all__ = [
    "IntegratedDonationPipeline",
    "JivaharAIFacade"
]
