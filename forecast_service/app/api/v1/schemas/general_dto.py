from enum import Enum
from pydantic import BaseModel


class ForecastResponse(BaseModel):
    """Модель ответа на прогноз. Всегда 3 месяца"""
    
    month_1: float 
    month_2: float
    month_3: float


class ConfidenceIntervalEnum(str, Enum):
    """Доверительный интервал"""

    low = '90%'
    medium = '95%'
    hight = '99%'
