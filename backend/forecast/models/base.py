from backend.forecast.datalake_service import DataLakeDTO

from typing import List, Tuple, Any
from abc import ABC, abstractmethod


class BaseForecast(ABC):
    """Интерфейс предсказания индекса"""
    
    @abstractmethod
    def __init__(self, **hparams) -> None:
        ...
    
    @property
    @abstractmethod
    def GetRequiredFeatures(self) -> list[str]:
        """Получить список всех необходимых признаков"""
        ... 
    
    @abstractmethod
    def set_data(self, data: DataLakeDTO) -> None:
        ...
    
    @abstractmethod
    def train_model(self) -> None:
        ...
    
    @abstractmethod
    def predict(self) -> list[tuple[float]]:
        ...   
    
    
    