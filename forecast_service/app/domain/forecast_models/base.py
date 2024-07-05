from typing import Union
from abc import ABC, abstractmethod


Params = dict[str, Union[float, int]]

class BaseForecast(ABC):
    """Интерфейс предсказания индекса"""
    
    @abstractmethod
    def __init__(self, **hparams: Params) -> None:
        ...
        
    @abstractmethod
    def set_data(self) -> "BaseForecast":
        ...
    
    @abstractmethod
    def preprocess_features(self) -> "BaseForecast":
        ...
    
    @abstractmethod
    def train(self) -> "BaseForecast":
        ...
    
    @abstractmethod
    def predict(self) -> tuple[float]:
        ...   
    
    
    