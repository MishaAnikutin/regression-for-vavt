from pydantic import BaseModel, Field
from app.schemas.ml.params import RNNHyperparameters, CatBoostHyperparameters
from app.schemas.ml.features import Feature, IPPFeatures
from app.schemas.common import ConfidenceIntervalEnum


class FeatureRequest(BaseModel):
    index_name: str = Field(
        default="Индекс промышленного производства",
        description="Название индекса (временного ряда) для которого хотите получить список признаков для модели"
    )


class BaseRequest(BaseModel):
    """
    DTO для базового запроса на прогноз временного ряда с помощью рекурентных нейронных сетей в архитектуре LSTM

    Параметры:
    - hparams:             гиперпараметры модели
    - target:              временной ряд индекса, который предсказываем
    """

    hparams: RNNHyperparameters
    target: Feature = Field(None, description="Переменная для предсказания")


class IPPRequestCB(BaseModel):
    """
    DTO для запроса на прогноз индекса промышленного производства производства на CatBoost Regressor

    Параметры:
    - hparams:             гиперпараметры CatBoost
    - confidence_interval: доверительный интервал
    - goal:                временной ряд индекса, который предсказываем
    - features:            временной ряд признаков индекса
    """

    hparams: CatBoostHyperparameters
    confidence_interval: ConfidenceIntervalEnum

    ipp: Feature = Field(None, description="Индекс промышленного производства")
    features: IPPFeatures


class IPPRequestRNN(BaseModel):
    """
    DTO для запроса на прогноз индекса промышленного производства производства по рекурентной нейронной сети

    Параметры:
    - hparams:             гиперпараметры RNN
    - confidence_interval: доверительный интервал
    - goal:                временной ряд индекса, который предсказываем
    - features:            временной ряд признаков индекса
    """

    hparams: RNNHyperparameters
    confidence_interval: ConfidenceIntervalEnum

    ipp: Feature = Field(None, description="Индекс промышленного производства")
    features: IPPFeatures
