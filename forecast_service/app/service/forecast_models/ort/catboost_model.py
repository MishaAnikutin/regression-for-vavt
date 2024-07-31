import pandas as pd
from datetime import date, timedelta
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from app.service.data_preprocess import TimeSeries
from app.domain.forecast_interface import BaseForecast
from app.schemas import Feature, ForecastResponse, ModelScore, CatBoostHyperparameters


class ORTForecast(BaseForecast):
    """Модель прогноза Оборота розничной торговли на CatBoost"""

    model_1_features = ('ort_lag_2', 'salary_lag_1', 'news_lag_1', 'business_clim_lag_1')
    model_2_features = ('ort_lag_3', 'salary_lag_2', 'news_lag_2', 'business_clim_lag_2')
    model_3_features = ('salary_lag_3', 'news_lag_3', 'business_clim_lag_3')

    def __init__(self, hparams: CatBoostHyperparameters):
        self._hparams = dict(hparams)

        self._model_1 = CatBoostRegressor(**self._hparams, verbose=False)
        self._model_2 = CatBoostRegressor(**self._hparams, verbose=False)
        self._model_3 = CatBoostRegressor(**self._hparams, verbose=False)

        self._raw_data = None
        self._df = None

    def set_data(self, ort: Feature, news: Feature, salary: Feature, business_clim: Feature) -> "ORTForecast":

        self._raw_data = dict(
            ort=TimeSeries(ort.values, ort.dates),
            news=TimeSeries(news.values, news.dates),
            salary=TimeSeries(salary.values, salary.dates),
            business_clim=TimeSeries(business_clim.values, business_clim.dates)
        )

        return self

    def preprocess_features(self) -> "ORTForecast":
        # делаем копию self._raw_data
        _raw_data = {k: v for k, v in self._raw_data.items()}

        df = pd.DataFrame({'date': []})

        for feature, columns in _raw_data.items():
            feature_df = pd.DataFrame({'date': columns.dates, feature: columns.values})

            # приводим даты к формату: год-месяц-1, чтобы их обьеденить по дате верно
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

        for feature in features:
            for i in range(1, 3 + 1):
                df[f'{feature}_lag_{i}'] = df[feature].shift(i)

        df['ort_lag_1'] = df['ort'].shift(1)
        df['ort_lag_2'] = df['ort'].shift(2)
        df['ort_lag_3'] = df['ort'].shift(3)

        self._df = (df.sort_values(by='date')
                    .dropna())

        return self

    def _features_filter(self, model_features: tuple):
        return self._df.loc[:, [*model_features]]

    @staticmethod
    def _train(model, X, y):
        model.fit(X=X, y=y)

        return model

    def train(self) -> "ORTForecast":
        self._model_1 = self._train(self._model_1, X=self._features_filter(self.model_1_features), y=self._df['ort'])
        self._model_2 = self._train(self._model_2, X=self._features_filter(self.model_2_features), y=self._df['ort_lag_1'])
        self._model_3 = self._train(self._model_3, X=self._features_filter(self.model_3_features), y=self._df['ort_lag_2'])

        return self

    def _score(self, x, y) -> ModelScore:
        predict = self._model_1.predict(x)

        return ModelScore(
            mape=mean_absolute_percentage_error(y, predict),
            r2_score=r2_score(y, predict)
        )

    def _iteration_predict(self, data):
        return (
            self._model_1.predict(data),
            self._model_2.predict(data),
            self._model_3.predict(data),
        )

    def predict(self) -> ForecastResponse:
        x = self._df.loc[:, [*self.model_1_features]]

        # Предсказание модели предыдущих значений
        previous = self._model_1.predict(x)

        # Предсказанеи последующих 3 по последнему значению
        predict = self._iteration_predict(x.iloc[-1, ])

        # Получаем score
        scores = [self._score(x=x, y=self._df.ort)]

        return ForecastResponse(
            previous=previous,
            predict=predict,
            scores=scores
        )
