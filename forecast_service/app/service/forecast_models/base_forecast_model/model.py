import numpy as np
import pandas as pd
from datetime import date
from functools import lru_cache
from typing import Iterable

from keras.src import Sequential
from keras.src.optimizers import Adam
from keras.src.layers import GRU, Dense, Dropout
from keras.src.metrics import MeanAbsolutePercentageError
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from app.domain.forecast_interface import BaseForecast
from app.schemas import RNNHyperparameters, Feature, ForecastResponse, ModelScore


FData = list[float]
IData = list[int]


class BaseForecastService(BaseForecast):
    """Сервис для прогноза временных рядов с помощью рекурентных нейронных сетей"""
    
    forecast_horizon = 1
    last_day = date(year=2015, month=1, day=1)
    
    def __init__(
            self,
            hparams: RNNHyperparameters
    ):
        self._df = None
        self._scaler = None
        self._hparams = hparams
        self._model = self._init_model()

    def _init_model(self):
        model = Sequential()

        for _ in range(self._hparams.n_layers - 1):
            # n-1 раз добавляем слой с return_sequences=True
            model.add(GRU(units=self._hparams.units, return_sequences=True, input_shape=(self._hparams.lookback, 1)))
            # Добавляем слой для регуляризации
            model.add(Dropout(0.1))

        # на последнем добавляем return_sequences=False
        model.add(GRU(units=self._hparams.units, return_sequences=False, input_shape=(self._hparams.lookback, 1)))
        model.add(Dropout(0.1))

        # Прогноз всегда на 1 месяц. На 2 и 3 будут итерационно
        model.add(Dense(units=self.forecast_horizon, activation='linear'))

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=[MeanAbsolutePercentageError()]
        )

        return model

    def set_data(self, target_data: Feature) -> "BaseForecastService":
        self._df = pd.DataFrame({'date': target_data.dates, 'goal': target_data.values})

        return self
                
    def preprocess_features(self) -> "BaseForecastService":
        # сделаем даты из строк в datetime.date
        self._df.date = pd.to_datetime(self._df.date, format='%d.%m.%Y').dt.date

        # Очищаем данные с критической точки (обычно начало 2015 года) и сортируем по дате
        # Важно чтобы первые значения были ~2015 а последние уже наши дни
        # self._df = self._df[self._df.date > self.last_day].sort_values(by='date')
        self._df = self._df.sort_values(by='date')

        # создаем скалер
        self._scaler: StandardScaler = StandardScaler().fit(self._df[['goal']])

        # создаем z_goal которая является скалированной goal
        self._df['z_goal'] = self._scaler.transform(self._df[['goal']])

        return self
    
    def _reshape_data_to_model(self, data: Iterable) -> tuple[np.array, np.array]:
        # создаем период времени, начиная с lookback (на сколько назад смотрим) до конца, без длины прогноза
        data_range = range(self._hparams.lookback, len(data) - self.forecast_horizon)

        x = np.array([data[i - self._hparams.lookback:i] for i in data_range])
        x = np.reshape(x, (x.shape[0], self._hparams.lookback, 1))

        y = np.array([data[i:i + self.forecast_horizon] for i in data_range])

        return x, y

    def _train(self, model, batch_size, epochs):
        x, y = self._reshape_data_to_model(self._df['z_goal'])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

        model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            batch_size=batch_size,
            epochs=epochs    
        )
        
        return model
    
    def train(self) -> "BaseForecastService":
        # TypeError: unhashable type: 'numpy.ndarray'
        self._model = self._train(self._model, self._hparams.batch_size, self._hparams.epochs)
                
        return self
    
    @staticmethod
    def _score(model, x, y) -> ModelScore:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        
        y_pred = model.predict(x_test)

        return ModelScore(
            mape=mean_absolute_percentage_error(y_test, y_pred),
            r2_score=r2_score(y_test, y_pred)
        )

    @staticmethod
    def _x_update_and_reshape(lookback, x_prev, y_pred=None):
        if y_pred is None:
            # Первая итерация, значит получили все значения
            return x_prev[-1]

        # убираем из массива первый элемент и добавляем последний прогноз
        return np.reshape(
            np.append(x_prev[1:], y_pred[-1]),
            (lookback, 1)
        )

    def _one_predict_iteration(self, model, lookback, x, y=None):
        x = self._x_update_and_reshape(lookback, x, y)

        return x, model.predict(x)

    # @lru_cache(maxsize=8)
    def _iteration_predict(self, model, z_goal):
        x, _ = self._reshape_data_to_model(data=z_goal)

        predictions = list()
        new_x_list = [x]

        # делаем прогноз на 3 дня
        for _ in range(3):

            x = new_x_list[-1]
            y = None if len(predictions) == 0 else predictions[-1]

            new_x, predict = self._one_predict_iteration(model, self._hparams.lookback, x, y)

            predictions.append(predict[-1])
            new_x_list.append(new_x)

        return predictions

    def predict(self) -> ForecastResponse:
        # Предсказание модели предыдущих значений
        previous = list(self._scaler.inverse_transform((self._model.predict(self._df.z_goal[self._hparams.lookback:]))))

        # Предсказанеи последующих 3
        month_1, month_2, month_3 = self._scaler.inverse_transform(self._iteration_predict(self._model, self._df.z_goal))

        return ForecastResponse(
            previous=previous,
            month_1=month_1,
            month_2=month_2,
            month_3=month_3,
            scores=[]
        )
