from typing import Union
from catboost import CatBoostRegressor

from app.domain.forecast_models import BaseForecast


FData = list[float]
IData = list[int]


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
    
    def __init__(self, hparams: dict[str, Union[FData, IData]]) -> None:
        # прогноз на 1 месяц 
        self.__model_1 = CatBoostRegressor(**hparams, verbose = False) 
        
        # прогноз на 2 месяца
        self.__model_2 = CatBoostRegressor(**hparams, verbose = False) 
        
        # прогноз на 3 месяца
        self.__model_3 = CatBoostRegressor(**hparams, verbose = False)

    
    def set_data(
            self,
            goal: FData,
            news: FData, 
            consumer_price: FData,
            exchange_rate: FData,
            cb_monitor: FData,
            bussines_clim: FData,
            curs: FData,
            rzd: FData
        ) -> None:
        
        self.goal = goal 
        self.news = news
        self.consumer_price = consumer_price
        self.exchange_rate = exchange_rate
        self.cb_monitor = cb_monitor
        self.bussines_clim = bussines_clim 
        self.curs = curs
        self.rzd = rzd
        
        self._preprocess_features()
        
    def _preprocess_features(self):
        """Предобработка признаков. Создадим переменные с лагом"""
        ...
    
    def train(self):
        print(1)
        
    def predict(self):
        return 1, 2, 3
