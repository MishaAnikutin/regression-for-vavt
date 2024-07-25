from typing import Union, TypeAlias

from pydantic import BaseModel, Field


class CatBoostHyperparameters(BaseModel):
    depth: int = Field(default=3)
    learning_rate: float = Field(default=0.1)
    l2_leaf_reg: float = Field(default=.005)
    iterations: int = Field(default=8)


class RNNHyperparameters(BaseModel):
    """Базовая модель для RNN"""

    lookback: int = Field(default=6, description="Сколько берем данных для прогноза")
    horizon: int = Field(default=3, description="На сколько прогнозируем")
    units: int = Field(default=2, description="Количество нейронов в слое")
    n_layers: int = Field(default=2, description="Количество слоев")
    batch_size: int = Field(default=5, description="")
    epochs: int = Field(default=100, description="")
    learning_rate: float = Field(default=0.0001, description="Шаг обучения")


class NHiTSHyperparameters(BaseModel):
    """Базовая модель для RNN"""

    lookback: int = Field(default=6, description="Сколько берем данных для прогноза")
    horizon: int = Field(default=3, description="На сколько прогнозируем")
    epochs: int = Field(default=100, description="Количество эпох обучения")
    learning_rate: float = Field(default=0.0001, description="Шаг обучения")


BaseHyperparameters: TypeAlias = Union[RNNHyperparameters, CatBoostHyperparameters, NHiTSHyperparameters]
