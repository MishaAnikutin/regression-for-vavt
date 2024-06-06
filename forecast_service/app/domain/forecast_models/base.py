from typing import Union
from abc import ABC, abstractmethod


Params = dict[str, Union[float, int]]

class BaseForecast(ABC):
    """Интерфейс предсказания индекса"""
    
    @abstractmethod
    def __init__(self, **hparams: Params) -> None:
        ...
        
    @abstractmethod
    def set_data(self) -> None:
        ...
    
    @abstractmethod
    def train_model(self) -> None:
        ...
    
    @abstractmethod
    def predict(self) -> tuple[float]:
        ...   
    
    
    