from typing import Union, TypeAlias

from pydantic import BaseModel, Field, conint, confloat


class CatBoostHyperparameters(BaseModel):
    depth: conint(gt=0, lt=32) = Field(default=3)
    learning_rate: confloat(gt=0, lt=1) = Field(default=0.1)
    l2_leaf_reg: confloat(gt=0, lt=5) = Field(default=.005)
    iterations: conint(gt=0, lt=32) = Field(default=8)


class RNNHyperparameters(BaseModel):
    """Базовая модель для RNN"""

    lookback: conint(gt=0, lt=24) = Field(default=6, description="Сколько берем данных для прогноза")
    horizon: conint(gt=0, lt=7) = Field(default=3, description="На сколько прогнозируем")
    units: conint(gt=0, lt=10) = Field(default=2, description="Количество нейронов в слое")
    n_layers: conint(gt=0, lt=10) = Field(default=2, description="Количество слоев")
    batch_size: conint(gt=0, lt=500) = Field(default=5, description="")
    epochs: conint(gt=0, lt=500) = Field(default=100, description="")
    learning_rate: confloat(gt=0, lt=1) = Field(default=0.0001, description="Шаг обучения")


class NHiTSHyperparameters(BaseModel):
    """Базовая модель для RNN"""

    lookback: conint(gt=0, lt=24) = Field(default=6, description="Сколько берем данных для прогноза")
    horizon: conint(gt=0, lt=13) = Field(default=3, description="На сколько прогнозируем")
    epochs: conint(gt=0, lt=500) = Field(default=100, description="Количество эпох обучения")
    learning_rate: confloat(gt=0, lt=1) = Field(default=0.0001, description="Шаг обучения")


BaseHyperparameters: TypeAlias = Union[RNNHyperparameters, CatBoostHyperparameters, NHiTSHyperparameters]
