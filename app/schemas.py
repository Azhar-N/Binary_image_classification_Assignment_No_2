"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class PredictionResponse(BaseModel):
    label: str          # "cat" or "dog"
    confidence: float   # Probability of predicted class (0.0 - 1.0)
    cat_probability: float
    dog_probability: float
