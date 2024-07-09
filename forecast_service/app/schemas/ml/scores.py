from enum import Enum
from pydantic import BaseModel


class ModelScore(BaseModel):
    mape: float
    r2_score: float 
