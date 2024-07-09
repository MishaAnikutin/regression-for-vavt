from abc import ABC, abstractmethod

from app.schemas import Feature, BaseHyperparameters, ForecastResponse


class BaseForecast(ABC):
    """Интерфейс предсказания индекса"""

    @abstractmethod
    def __init__(self, **hparams: BaseHyperparameters) -> None:
        ...

    @abstractmethod
    def set_data(self, *dataseries: Feature) -> "BaseForecast":
        ...

    @abstractmethod
    def preprocess_features(self) -> "BaseForecast":
        ...

    @abstractmethod
    def train(self) -> "BaseForecast":
        ...

    @abstractmethod
    def predict(self) -> ForecastResponse:
        ...
