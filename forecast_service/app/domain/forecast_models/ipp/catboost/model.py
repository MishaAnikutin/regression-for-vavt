from typing import Union, Callable
from catboost import CatBoostRegressor
import numpy as np

from app.domain.forecast_models import BaseForecast


FData = list[float]
IData = list[int]



def print_result(func: Callable):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f'{func.__name__} = {result}\t{args=}\t{kwargs=}\n')
        return result
    return wrapper


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
        "interest_rate": 4,  # Ключевая ставка 
        "cb_monitor": 2,     # Промышленность. Как изменился спрос на продукцию, товары, услуги?
        "bussines_clim": 1,  # Индекс бизнес климата
        "curs": 0,           # Курс рубля к доллару
        "rzd": 1             # Поставки РЖД
    }
    
    model_1_features = ('goal_lag_1', 'goal_lag_2', 'news_lag_1', 'news_lag_2', 'news_lag_3', 'cb_monitor_lag_2', 'bussines_clim_lag_1', 'rzd_lag_1', 'interest_rate_lag_4', 'consumer_price', 'curs', )        
    model_2_features = ('goal_lag_2', 'news_lag_2', 'news_lag_3', 'news_lag_4', 'cb_monitor_lag_3', 'bussines_clim_lag_2', 'rzd_lag_2', 'interest_rate_lag_5', 'consumer_price_lag_1', 'curs_lag_1', )
    model_3_features = ('goal_lag_3', 'news_lag_3', 'news_lag_4', 'news_lag_5', 'cb_monitor_lag_4', 'bussines_clim_lag_3', 'rzd_lag_3', 'interest_rate_lag_6', 'consumer_price_lag_2', 'curs_lag_2',)
    
    # Сохранение предыдущих данных и гиперпараметров
    prev_features = None
    prev_args = None
    prev_model_1 = None 
    prev_model_2 = None 
    prev_model_3 = None 

    
    def __init__(self, hparams: dict[str, Union[FData, IData]]) -> None:
        self.args = hparams
        
        # прогноз на 1 месяц 
        self.__model_1 = CatBoostRegressor(**hparams, verbose=False) 
        
        # прогноз на 2 месяца
        self.__model_2 = CatBoostRegressor(**hparams, verbose=False) 
        
        # прогноз на 3 месяца
        self.__model_3 = CatBoostRegressor(**hparams, verbose=False)

    
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
        
        self.goal = np.array(goal)
        
        self.features = {
            "news": news,
            "consumer_price": consumer_price,
            "interest_rate": exchange_rate,
            "bussines_clim": bussines_clim,
            "cb_monitor": cb_monitor,
            "curs": curs,
            "rzd": rzd
        }
                
        self._preprocess_features()
                
    def _preprocess_features(self):
        """Предобработка признаков. Создадим переменные с лагом"""
        
        self.features = {key: np.array(data) for key, data in self.features.items()}
        features_with_lag = dict()
        
        for i in range(6):
            for feature in self.features.keys():
                lag = self.feature_lags[feature]
                data = self.features[feature]
                
                features_with_lag[f'{feature}_lag_{i + lag}'] = self.shift(data, i)

        features_with_lag['goal_lag_1'] = self.shift(self.goal, 1) ** 2
        features_with_lag['goal_lag_2'] = self.shift(self.goal, 2) ** 2
        features_with_lag['goal_lag_3'] = self.shift(self.goal, 3) ** 2

        self.goal_lag_1 = features_with_lag['goal_lag_1']
        self.goal_lag_2 = features_with_lag['goal_lag_2']
        self.goal_lag_3 = features_with_lag['goal_lag_3']
        
        self.features = dict(list(features_with_lag.items()) + list(self.features.items()))

    @staticmethod
    def shift(data: list, i: int) -> np.ndarray:
        """Сдвигает на i единиц элементы в списке 
        
        при i = 2: [1, 2, 3, 4, 5, 6] -> [1, 1, 1, 2, 3, 4]
        
        nan заполняется еще последним значением

        Args:
            data (list): список данных
            i (int): размер сдвига
        """
        
        if i == 0:
            return data
        return np.concatenate((np.full(i, data[0]), data[:-i]))
    
    def _filter_data(self, model_features: tuple):
        return np.array([self.features[key] for key in model_features]).transpose()

    
    def train(self):
        if self.prev_features == self.features and self.prev_args == self.args:
            # TODO: так не работает
            return self._set_models_from_cache() 
        
        self.__model_1.fit(X=self._filter_data(self.model_1_features), y=self.goal)
        self.__model_2.fit(X=self._filter_data(self.model_2_features), y=self.goal_lag_1)        
        self.__model_3.fit(X=self._filter_data(self.model_3_features), y=self.goal_lag_2)    
        
        self._cache_args_and_features()
        self._cache_models()
        
    def _cache_args_and_features(self, cache={}):
        self.prev_args = {k: v for k, v in self.args.items()}
        self.prev_features = {k: v for k, v in  self.features.items()}
    
    def _cache_models(self):
        self.prev_model_1 = self.__model_1
        self.prev_model_2 = self.__model_2
        self.prev_model_3 = self.__model_3
        
    def _set_models_from_cache(self):
        self.__model_1 = self.prev_model_1
        self.__model_2 = self.prev_model_2
        self.__model_3 = self.prev_model_3

    def predict(self):
        data = self._filter_data(self.model_1_features)[-1]
                
        predict_1 = self.__model_1.predict(data)
        predict_2 = self.__model_2.predict(data) ** 0.5
        predict_3 = self.__model_3.predict(data) ** 0.5
                
        return predict_1, predict_2, predict_3
