from typing import List, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np 


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
    def set_data(self, index_data: np.ndarray, feature_data: np.ndarray[np.ndarray[float]]) -> None:
        ...
    
    @abstractmethod
    def train_model(self) -> None:
        ...
    
    @abstractmethod
    def predict(self) -> list[tuple[float]]:
        ...   
    
    
    