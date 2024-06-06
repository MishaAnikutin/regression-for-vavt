from typing import Union

import numpy as np 
from pandas import DataFrame
from catboost import CatBoostRegressor 

from backend.forecast.models.base import BaseForecast 
from backend.forecast.datalake_service import IPPDTO


class IPPForecast(BaseForecast):
    """
    Модель прогноза Индекса промышленного производства на CatBoost
    
    При предобработке данных, разные признаки мы будем сдвигать по определенному лагу
    
    Условно, шоки поставок РЖД повлияют на экономику спустя только несколько
    месяцев
    
    Поэтому, будем хранить словарь с маппингом столбцов и их лаггом 
    """
    
    feature_lags = {
        "news": 1,           # Новостной индекс ЦБ
        "consumer_price": 0, # Индекс цен на электроэнергию в первой ценовой зоне
        "exchange_rate": 6,  # Ключевая ставка 
        "cb_monitor": 2,     # Промышленность. Как изменился спрос на продукцию, товары, услуги?
        "bussines_clim": 2,  # Индекс бизнес климата
        "curs": 0,           # Курс рубля к доллару
        "rzd": 1             # Поставки РЖД
    }
    
    def __init__(self, hparams: dict[str, Union[int, float]]) -> None:
        self.__model_1 = CatBoostRegressor(**hparams, verbose = False) # 1 месяц прогноз
        self.__model_2 = CatBoostRegressor(**hparams, verbose = False) # 2 месяца прогноз
        self.__model_3 = CatBoostRegressor(**hparams, verbose = False) # 3 месяца прогноз

    
    def set_data(self, data: IPPDTO.Data):
        self.goal = data.ipp  
        self.news = data.news
        self.consumer_price = data.consumer_price
        self.exchange_rate = data.exchange_rate
        self.cb_monitor = data.cb_monitor
        
    
    
    @staticmethod
    def _preprocess_features(feature_data):
        """Предобработка признаков

        В начале сдвинем 
        """
        
    