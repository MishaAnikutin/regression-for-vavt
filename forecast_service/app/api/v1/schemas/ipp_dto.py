from pydantic import BaseModel, Field, field_validator
from datetime import date, datetime

from .catboost_dto import CatBoostHyperparameters
from .general_dto import ConfidenceIntervalEnum


class TimeSeriesData(BaseModel):
    values: list[float]
    dates: list[str]


class IPPFeatures(BaseModel):
    news:           TimeSeriesData = Field(None, description="Новостной идекс ЦБ, Россия")
    cb_monitor:     TimeSeriesData = Field(None, description="Промышленность. Как изменился спрос на продукцию, товары, услуги?") 
    bussines_clim:  TimeSeriesData = Field(None, description="Промышленность Индикатор бизнес-климата Банка России") 
    interest_rate:  TimeSeriesData = Field(None, description="Ключевая ставка ")
    rzd:            TimeSeriesData = Field(None, description="Погрузка на сети РЖД")
    consumer_price: TimeSeriesData = Field(None, description="Индекс цен на электроэнергию в первой ценовой зоне")
    curs:           TimeSeriesData = Field(None, description="Курс рубля к доллару США")


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
        
        ipp: TimeSeriesData = Field(None, description="Индекс промышленного производства")
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
        
        hparams: str
        confidence_interval: ConfidenceIntervalEnum
        
        ipp: TimeSeriesData = Field(None, description="Индекс промышленного производства")
        features: IPPFeatures
