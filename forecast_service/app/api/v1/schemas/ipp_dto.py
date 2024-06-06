from pydantic import BaseModel, Field

from .catboost_dto import CatBoostHyperparameters
from .general_dto import ConfidenceIntervalEnum

    
class IPPFeatures(BaseModel):   
    news:           list[float] = Field(None, description="Новостной индекс ЦБ")
    cb_monitor:     list[float] = Field(None, description="Промышленность. Как изменился спрос на продукцию, товары, услуги?") 
    bussines_clim:  list[float] = Field(None, description="Промышленность Индикатор бизнес-климата Банка России") 
    exchange_rate:  list[float] = Field(None, description="Ключевая ставка ")
    rzd:            list[float] = Field(None, description="Погрузка на сети РЖД")
    consumer_price: list[float] = Field(None, description="Индекс цен на электроэнергию в первой ценовой зоне")
    curs:           list[float] = Field(None, description="Курс рубля к доллару США")


class IPPRequest(BaseModel):
        """
        DTO для запроса на прогноз индекса промышленного производства производства
        
        Параметры:
        - hparams:             гиперпараметры CatBoost
        - confidence_interval: доверительный интервал
        - goal:                временной ряд индекса, который предсказываем
        - features:            временной ряд признаков индекса
        """
        
        hparams: CatBoostHyperparameters
        confidence_interval: ConfidenceIntervalEnum
        
        ipp: list[float] = Field(None, description="Индекс промышленного производства")
        features: IPPFeatures
