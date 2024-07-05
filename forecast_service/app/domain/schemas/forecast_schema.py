from dataclasses import dataclass


@dataclass
class ModelScoreDTO:
    mape: float
    r2_score: float


@dataclass
class ForecastDTO:
    month_1: float
    month_2: float
    month_3: float
    
    scores: list[ModelScoreDTO]
