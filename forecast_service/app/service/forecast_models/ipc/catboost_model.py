import pandas as pd
from datetime import date
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from app.service.data_preprocess import TimeSeries
from app.domain.forecast_interface import BaseForecast
from app.schemas import Feature, ForecastResponse, ModelScore, CatBoostHyperparameters


class IPCForecast(BaseForecast):
    """
    Модель прогноза Индекса промышленного производства на CatBoost

    При предобработке данных, разные признаки мы будем сдвигать по определенному лагу

    Условно, шоки поставок РЖД повлияют на экономику спустя только несколько
    месяцев

    Поэтому, будем хранить словарь с маппингом столбцов и их лаггом
    """

    # Дата начала отчета для данных
    date_start = date(year=2015, month=1, day=1)

    model_1_features = ('curs', 'interest_rate', 'ipc_lag_1', 'share_m0')
    model_2_features = ('curs_lag_1', 'interest_rate_lag_1', 'ipc_lag_2', 'share_m0_lag_1')
    model_3_features = ('curs_lag_2', 'interest_rate_lag_2', 'ipc_lag_3', 'share_m0_lag_2')

    def __init__(
            self,
            hparams: CatBoostHyperparameters,
    ):
        "Конструктор класса. Класс будет неизменяемым, поэтому все признаки пересоздаются"
        self._hparams = dict(hparams)

        self._model_1 = CatBoostRegressor(**self._hparams, verbose=False)
        self._model_2 = CatBoostRegressor(**self._hparams, verbose=False)
        self._model_3 = CatBoostRegressor(**self._hparams, verbose=False)

        self._raw_data = None
        self._df = None

    def set_data(
            self,
            ipc: Feature,
            curs: Feature,
            interest_rate: Feature,
            agg_m0: Feature,
            money_supply: Feature
    ) -> "IPCForecast":
        self._raw_data = {
            "ipc": TimeSeries(ipc.values, ipc.dates),
            "curs": TimeSeries(curs.values, curs.dates),
            "interest_rate": TimeSeries(interest_rate.values, interest_rate.dates),
            "agg_m0": TimeSeries(agg_m0.values, agg_m0.dates),
            "money_supply": TimeSeries(money_supply.values, money_supply.dates)
        }

        return self

    @staticmethod
    def _get_share_m0(agg_m0: TimeSeries, money_supply: TimeSeries):
        # создаем датафреймы
        agg_m0_df = pd.DataFrame({'dates': agg_m0.dates, 'agg_m0': agg_m0.values})
        money_supply_df = pd.DataFrame({'dates': money_supply.dates, 'money_supply': money_supply.values})

        # обьединяем данные денежного аггрегата M0 и денежной массы
        df = agg_m0_df.merge(money_supply_df, on='dates', how='outer')

        df['values'] = df['agg_m0'] / df['money_supply']
        return TimeSeries(dates=df.dates, values=df['values'])

    def preprocess_features(self) -> "IPCForecast":
        """Предобработка признаков. Создадим переменные с лагом"""

        # делаем копию self._raw_data
        _raw_data = {k: v for k, v in self._raw_data.items()}

        _raw_data['curs'] = _raw_data['curs'].days_to_months()
        _raw_data['money_supply'] = _raw_data['money_supply'].days_to_months()
        _raw_data['agg_m0'] = _raw_data['agg_m0'].days_to_months()

        _raw_data['share_m0'] = self._get_share_m0(_raw_data['agg_m0'], _raw_data['money_supply'])

        preprocessed_features = ('curs', 'money_supply', 'agg_m0', 'share_m0')

        df = pd.DataFrame({'date': []})

        for feature, columns in _raw_data.items():
            feature_df = pd.DataFrame({'date': columns.dates, feature: columns.values})

            # если мы их еще не предобрабатывали
            if feature not in preprocessed_features:
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

        df['ipc_lag_1'] = df['ipc'].shift(1) ** 2
        df['ipc_lag_2'] = df['ipc'].shift(2) ** 2
        df['ipc_lag_3'] = df['ipc'].shift(3) ** 2

        for feature in features:
            for i in range(1, 3 + 1):
                df[f'{feature}_lag_{i}'] = df[feature].shift(i)

        self._df = (df[~pd.isna(df.ipc)]
                    [df.date >= self.date_start]
                    .sort_values(by='date')
                    .bfill()
                    .dropna())

        return self

    def _features_filter(self, model_features: tuple):
        return self._df.loc[:, [*model_features]]

    def train(self) -> "IPCForecast":
        self._model_1.fit(X=self._features_filter(self.model_1_features), y=self._df['ipc'])
        self._model_2.fit(X=self._features_filter(self.model_2_features), y=self._df['ipc_lag_1'])
        self._model_3.fit(X=self._features_filter(self.model_3_features), y=self._df['ipc_lag_2'])

        return self

    @staticmethod
    def _score(model, x, y) -> ModelScore:
        predict = model.predict(x)

        return ModelScore(
            mape=mean_absolute_percentage_error(y, predict),
            r2_score=r2_score(y, predict)
        )

    def _iteration_predict(self, data):
        return (
            self._model_1.predict(data),
            self._model_2.predict(data) ** 0.5,
            self._model_3.predict(data) ** 0.5,
        )

    def predict(self) -> ForecastResponse:
        x = self._df.loc[:, [*self.model_1_features]]

        # Предсказание модели предыдущих значений
        previous = self._model_1.predict(x)

        # Предсказанеи последующих 3 по последнему значению
        predict = self._iteration_predict(x.iloc[-1, ])

        # Получаем score
        scores = [self._score(model=self._model_1, x=x, y=self._df.ipc)]

        return ForecastResponse(
            previous=previous,
            predict=predict,
            scores=scores
        )
