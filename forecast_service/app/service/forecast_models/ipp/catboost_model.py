import pandas as pd
from typing import Union
from datetime import date
from functools import lru_cache

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from app.service.data_preprocess import TimeSeries
from app.service.forecast_models import BaseForecast

from app.schemas import Feature, ForecastResponse, ModelScore, CatBoostHyperparameters


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
        "ipp": 0,             # Индекс промышленного производства
        "news": 1,            # Новостной индекс ЦБ
        "consumer_price": 0,  # Индекс цен на электроэнергию в первой ценовой зоне
        "interest_rate": 4,   # Ключевая ставка
        "cb_monitor": 2,      # Промышленность. Как изменился спрос на продукцию, товары, услуги?
        "bussines_clim": 1,   # Индекс бизнес климата
        "curs": 0,            # Курс рубля к доллару
        "rzd": 1              # Поставки РЖД
    }

    # Дата начала отчета для данных
    date_start = date(year=2015, month=1, day=1)

    model_1_features = (
        'ipp_lag_1', 'ipp_lag_2', 'news_lag_1', 'news_lag_2', 'news_lag_3', 'cb_monitor_lag_2', 'bussines_clim_lag_1',
        'rzd_lag_1', 'interest_rate_lag_4', 'consumer_price', 'curs',
    )

    model_2_features = (
        'ipp_lag_2', 'news_lag_2', 'news_lag_3', 'news_lag_4', 'cb_monitor_lag_3', 'bussines_clim_lag_2', 'rzd_lag_2',
        'interest_rate_lag_5', 'consumer_price_lag_1', 'curs_lag_1',
    )

    model_3_features = (
        'ipp_lag_3', 'news_lag_3', 'news_lag_4', 'news_lag_5', 'cb_monitor_lag_4', 'bussines_clim_lag_3', 'rzd_lag_3',
        'interest_rate_lag_6', 'consumer_price_lag_2', 'curs_lag_2',
    )

    def __init__(
            self,
            hparams: CatBoostHyperparameters,
            model_1=None,
            model_2=None,
            model_3=None,
            date_end=None,
            raw_data: dict[str, TimeSeries] = None,
            df=None
    ):
        "Конструктор класса. Класс будет неизменяемым, поэтому все признаки пересоздаются"
        self._args = dict(hparams)

        self._model_1 = CatBoostRegressor(**hparams, verbose=False) if model_1 is None else model_1
        self._model_2 = CatBoostRegressor(**hparams, verbose=False) if model_2 is None else model_2
        self._model_3 = CatBoostRegressor(**hparams, verbose=False) if model_3 is None else model_3

        self.date_end = date_end
        self._raw_data = raw_data
        self.df = df

    def set_data(
            self,
            ipp: Feature,
            news: Feature,
            consumer_price: Feature,
            interest_rate: Feature,
            cb_monitor: Feature,
            bussines_clim: Feature,
            curs: Feature,
            rzd: Feature
    ) -> "IPPForecast":

        raw_data = {
            "ipp": TimeSeries(ipp.values, ipp.dates),
            "news": TimeSeries(news.values, news.dates),
            "consumer_price": TimeSeries(consumer_price.values, consumer_price.dates),
            "interest_rate": TimeSeries(interest_rate.values, interest_rate.dates),
            "cb_monitor": TimeSeries(cb_monitor.values, cb_monitor.dates),
            "bussines_clim": TimeSeries(bussines_clim.values, bussines_clim.dates),
            "curs": TimeSeries(curs.values, curs.dates),
            "rzd": TimeSeries(rzd.values, rzd.dates)
        }

        day, month, year = map(int, ipp.dates[-1].split('.'))
        date_end = date(day=day, month=month, year=year)

        return IPPForecast(
            hparams=self._args, model_1=self._model_1, model_2=self._model_2, model_3=self._model_3,
            date_end=date_end, raw_data=raw_data
        )

    def preprocess_features(self) -> "IPPForecast":
        """Предобработка признаков. Создадим переменные с лагом"""

        _raw_data = {k: v for k, v in self._raw_data.items()}
        _raw_data['curs'] = _raw_data['curs'].days_to_months()

        df = pd.DataFrame({'date': []})

        for feature, columns in self._raw_data.items():
            feature_df = pd.DataFrame({'date': columns.dates, feature: columns.values})

            # приводим все даты к формату: ГОД-МЕСЯЦ-1 , чтобы их обьеденить по дате верно
            feature_df.date = feature_df.date.apply(
                lambda x: date(
                    year=int(x.split('.')[2]),
                    month=int(x.split('.')[1]),
                    day=1
                )
            )

            df = df.merge(feature_df, on='date', how='outer')

        df.date = pd.to_datetime(df.date, format='%d.%m.%Y').dt.date

        # столбцы динамически меняются, поэтому сохраняем и приводим к неизменяемому типу
        features = tuple(df.columns)

        for i in range(6):
            for feature in features:
                lag = self.feature_lags.get(feature, 0)

                df[f'{feature}_lag_{i + lag}'] = df[feature].shift(i + lag)

        df['ipp_lag_1'] = df['ipp_lag_1'] ** 2
        df['ipp_lag_2'] = df['ipp_lag_2'] ** 2
        df['ipp_lag_3'] = df['ipp_lag_3'] ** 2

        df = df \
            [~pd.isna(df.ipp)] \
            .sort_values(by='date') \
            .interpolate(method='linear') \
            .dropna()

        return IPPForecast(
            hparams=self._args, model_1=self._model_1, model_2=self._model_2, model_3=self._model_3,
            date_end=self.date_end, raw_data=_raw_data, df=df
        )

    def _features_filter(self, model_features: tuple):
        return self.df.loc[:, [*model_features]]

    @staticmethod
    @lru_cache(maxsize=8)
    def _train(model, X, y):
        model.fit(X=X, y=y)
        
        return model
    
    def train(self) -> "IPPForecast":
        model_1 = self._train(self._model_1, X=self._features_filter(self.model_1_features), y=self.df['ipp'])
        model_2 = self._train(self._model_2, X=self._features_filter(self.model_2_features), y=self.df['ipp_lag_1'])
        model_3 = self._train(self._model_3, X=self._features_filter(self.model_3_features), y=self.df['ipp_lag_2'])

        return IPPForecast(
            hparams=self._args, model_1=model_1, model_2=model_2, model_3=model_3,
            df=self.df, date_end=self.date_end, raw_data=self._raw_data
        )

    @staticmethod
    def _score(model, x, y) -> ModelScore:
        predict = model.predict(x)

        return ModelScore(
            mape=mean_absolute_percentage_error(y, predict),
            r2_score=r2_score(y, predict)
        )

    @staticmethod
    @lru_cache(maxsize=8)
    def _predict(model, data):
        return model.predict(data)

    def predict(self) -> ForecastResponse:
        # Берем самые последние данные, получаем предсказание и score для каждой модели

        data = self._features_filter(self.model_1_features).iloc[-1,]

        month_1 = self._predict(self._model_1, data)
        month_2 = self._predict(self._model_2, data) ** 0.5
        month_3 = self._predict(self._model_3, data) ** 0.5

        score_1 = self._score(self._model_1, self._features_filter(self.model_1_features), self.df.ipp)
        score_2 = self._score(self._model_2, self._features_filter(self.model_2_features), self.df.ipp_lag_1)
        score_3 = self._score(self._model_3, self._features_filter(self.model_3_features), self.df.ipp_lag_2)

        return ForecastResponse(
            month_1=month_1,
            month_2=month_2,
            month_3=month_3,
            scores=list[score_1, score_2, score_3]
        )
