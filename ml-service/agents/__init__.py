"""
Multi-agent pipeline package.

Agent execution order:
  IngestionAgent → FeatureAgent → ModelingAgent → PredictionAgent → ValidationAgent

Each agent accepts a typed context dict and returns an enriched context dict,
making the pipeline composable and testable in isolation.
"""
from agents.ingestion_agent import IngestionAgent
from agents.feature_agent import FeatureAgent
from agents.modeling_agent import ModelingAgent
from agents.prediction_agent import PredictionAgent
from agents.validation_agent import ValidationAgent

__all__ = [
    "IngestionAgent",
    "FeatureAgent",
    "ModelingAgent",
    "PredictionAgent",
    "ValidationAgent",
]
