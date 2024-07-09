from pydantic import BaseModel
from app.schemas.ml.scores import ModelScore

class ForecastResponse(BaseModel):
    """Модель ответа на прогноз на 3 месяца и качество модели"""

    month_1: float
    month_2: float
    month_3: float

    scores: list[ModelScore]


class FeatureResponse(BaseModel):
    dataset_uuid: str
    description: str


class FeaturesResponse(BaseModel):
    """Модель получения всех признаков для модели. Ответом будет словарь в формате признак: описание"""
    features: dict[str, FeatureResponse]
