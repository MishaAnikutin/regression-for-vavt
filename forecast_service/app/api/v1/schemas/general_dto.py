from enum import Enum
from pydantic import BaseModel


class ForecastResponse(BaseModel):
    """Модель ответа на прогноз на 3 месяца и качество модели"""
    month_1: float
    month_2: float 
    month_3: float

    scores: list
    

class ConfidenceIntervalEnum(str, Enum):
    """Доверительный интервал"""

    low = '90%'
    medium = '95%'
    hight = '99%'
