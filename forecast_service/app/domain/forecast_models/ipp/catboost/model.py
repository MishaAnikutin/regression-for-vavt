from sklearn.metrics import mean_absolute_percentage_error, r2_score
from catboost import CatBoostRegressor
from typing import Union, Callable
from functools import lru_cache
from datetime import date
import numpy as np
import pandas as pd

from app.domain.forecast_models import BaseForecast
from app.api.v1.schemas.ipp_dto import TimeSeriesData
from app.domain.data_preprocess import TimeSeries
from app.domain.schemas.forecast_schema import ForecastDTO, ModelScoreDTO

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
        "ipp": 0,            # Индекс промышленного производства
        "news": 1,           # Новостной индекс ЦБ
        "consumer_price": 0, # Индекс цен на электроэнергию в первой ценовой зоне
        "interest_rate": 4,  # Ключевая ставка 
        "cb_monitor": 2,     # Промышленность. Как изменился спрос на продукцию, товары, услуги?
        "bussines_clim": 1,  # Индекс бизнес климата
        "curs": 0,           # Курс рубля к доллару
        "rzd": 1             # Поставки РЖД
    }
    
    # Дата начала отчета для данных 
    date_start = date(year=2015, month=1, day=1)
    
    model_1_features = ('ipp_lag_1', 'ipp_lag_2', 'news_lag_1', 'news_lag_2', 'news_lag_3', 'cb_monitor_lag_2', 'bussines_clim_lag_1', 'rzd_lag_1', 'interest_rate_lag_4', 'consumer_price', 'curs', )        
    model_2_features = ('ipp_lag_2', 'news_lag_2', 'news_lag_3', 'news_lag_4', 'cb_monitor_lag_3', 'bussines_clim_lag_2', 'rzd_lag_2', 'interest_rate_lag_5', 'consumer_price_lag_1', 'curs_lag_1', )
    model_3_features = ('ipp_lag_3', 'news_lag_3', 'news_lag_4', 'news_lag_5', 'cb_monitor_lag_4', 'bussines_clim_lag_3', 'rzd_lag_3', 'interest_rate_lag_6', 'consumer_price_lag_2', 'curs_lag_2',)
    
    def __init__(self, hparams: dict[str, Union[FData, IData]]) -> None:
        self._args = hparams
        
        self._model_1 = CatBoostRegressor(**hparams, verbose=False) # прогноз на 1 месяц 
        self._model_2 = CatBoostRegressor(**hparams, verbose=False) # прогноз на 2 месяца
        self._model_3 = CatBoostRegressor(**hparams, verbose=False) # прогноз на 3 месяца
        
        self.df = None

    
    def set_data(
            self,
            ipp:            TimeSeriesData, 
            news:           TimeSeriesData,
            consumer_price: TimeSeriesData,
            interest_rate:  TimeSeriesData,
            cb_monitor:     TimeSeriesData,
            bussines_clim:  TimeSeriesData,
            curs:           TimeSeriesData,
            rzd:            TimeSeriesData
        ) -> "IPPForecast":
                
        self._raw_data = {
            "ipp":            TimeSeries(ipp.values, ipp.dates),
            "news":           TimeSeries(news.values, news.dates),
            "consumer_price": TimeSeries(consumer_price.values, consumer_price.dates),
            "interest_rate":  TimeSeries(interest_rate.values, interest_rate.dates),
            "cb_monitor":     TimeSeries(cb_monitor.values, cb_monitor.dates),
            "bussines_clim":  TimeSeries(bussines_clim.values, bussines_clim.dates),
            "curs":           TimeSeries(curs.values, curs.dates),
            "rzd":            TimeSeries(rzd.values, rzd.dates)
        }
                
        day, month, year = map(int, ipp.dates[-1].split('.'))
        self.date_end = date(day=day, month=month, year=year)
        
        return self 
                
    def preprocess_features(self) -> "IPPForecast":
        """Предобработка признаков. Создадим переменные с лагом"""

        self._raw_data['curs'] = self._raw_data['curs'].days_to_months()
                        
        self.df = pd.DataFrame({'date': []})
        
        for feature, columns in self._raw_data.items():
            feature_df = pd.DataFrame({'date': columns.dates, feature: columns.values})
            
            # приводим все даты к формату: ГОД-МЕСЯЦ-1 , чтобы их обьеденить по дате верно
            feature_df.date = feature_df.date.apply(
                lambda x: date(
                    year  = int(x.split('.')[2]),
                    month = int(x.split('.')[1]),
                    day   = 1
                )
            )
            
            self.df = self.df.merge(feature_df, on='date', how='outer')
        
        self.df.date = pd.to_datetime(self.df.date, format='%d.%m.%Y').dt.date

        # столбцы динамически меняются, поэтому сохраняем и приводим к неизменяемому типу
        features = tuple(self.df.columns)

        for i in range(6):
            for feature in features:
                lag = self.feature_lags.get(feature, 0)
                                
                self.df[f'{feature}_lag_{i + lag}'] = self.df[feature].shift(i + lag)

        self.df['ipp_lag_1'] = self.df['ipp_lag_1'] ** 2
        self.df['ipp_lag_2'] = self.df['ipp_lag_2'] ** 2
        self.df['ipp_lag_3'] = self.df['ipp_lag_3'] ** 2
        
        
        self.df = self.df\
            [~pd.isna(self.df.ipp)]\
            .sort_values(by='date')\
            .interpolate(method='linear')\
            .dropna()
        

                
        return self
    
    
    def _features_filter(self, model_features: tuple):
        return self.df.loc[:, [*model_features]]

    @lru_cache(maxsize=8)
    def train(self) -> "IPPForecast":
        self._model_1.fit(X=self._features_filter(self.model_1_features), y=self.df['ipp'])
        self._model_2.fit(X=self._features_filter(self.model_2_features), y=self.df['ipp_lag_1'])        
        self._model_3.fit(X=self._features_filter(self.model_3_features), y=self.df['ipp_lag_2'])
        
        return self
    
    @staticmethod
    def _score(model, x, y) -> ModelScoreDTO:
        predict = model.predict(x)

        return ModelScoreDTO(
            mape      = mean_absolute_percentage_error(y,predict),
            r2_score  = r2_score(y, predict)
        )
        
    def predict(self) -> ForecastDTO:
        # Берем самые последние данные, получаем предсказание и score для каждой модели
        
        data = self._features_filter(self.model_1_features).iloc[-1, ]

        month_1 = self._model_1.predict(data)
        month_2 = self._model_2.predict(data) ** 0.5
        month_3 = self._model_3.predict(data) ** 0.5
        
        score_1 = self._score(self._model_1, self._features_filter(self.model_1_features), self.df.ipp)
        score_2 = self._score(self._model_2, self._features_filter(self.model_2_features), self.df.ipp_lag_1)
        score_3 = self._score(self._model_3, self._features_filter(self.model_3_features), self.df.ipp_lag_2)

        return ForecastDTO(
            month_1=month_1,
            month_2=month_2,
            month_3=month_3,
            scores=[score_1, score_2, score_3]
        )
