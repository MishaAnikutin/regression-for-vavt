from typing import Union, TypeAlias

from pydantic import BaseModel, Field


class CatBoostHyperparameters(BaseModel):
    depth:         int = Field(default=3)
    learning_rate: float = Field(default=0.1)
    l2_leaf_reg:   float = Field(default=.005)
    iterations:    int = Field(default=8)


class RNNHyperparameters(BaseModel):
    """Базовая модель для RNN"""

    lookback:         int = Field(default=6, description="Сколько берем данных для прогноза")
    forecast_horizon: int = Field(default=3, description="На сколько делаем прогноз")
    units:            int = Field(default=1, description="Количество нейронов в слое")
    n_layers:         int = Field(default=1, description="Количество слоев")
    batch_size:       int = Field(default=64)
    dropout_rate:     float = Field(default=0.2)
    epochs:           int = Field(default=250)


BaseHyperparameters: TypeAlias = Union[RNNHyperparameters, CatBoostHyperparameters]
