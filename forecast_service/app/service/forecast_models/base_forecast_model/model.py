import numpy as np
import pandas as pd
from datetime import date
from functools import lru_cache
from typing import Iterable

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.layers import GRU, LSTM, BatchNormalization, Dropout, Dense, TimeDistributed

from app.service.forecast_models import BaseForecast

from app.schemas import RNNHyperparameters, Feature, ForecastResponse, ModelScore

FData = list[float]
IData = list[int]


class BaseForecastService(BaseForecast):
    """Сервис для прогноза временных рядов с помощью рекурентных нейронных сетей"""
    
    def __init__(
            self,
            hparams: RNNHyperparameters,
            model=None,
            df=None,
            scaler=None
    ):
        self._hparams = hparams
        self._model = model if model is not None else self._init_model(hparams)
        self._df = df
        self._scaler = scaler
        

    def _init_model(self, hparams: RNNHyperparameters):
        model = Sequential()

        for _ in range(hparams.n_layers - 1):
            model.add(LSTM(units=hparams.units, return_sequences=True, input_shape=(hparams.lookback, 1)))

        model.add(LSTM(units=hparams.units, return_sequences=False, input_shape=(hparams.lookback, 1)))
        model.add(Dense(units=hparams.forecast_horizon, activation='linear'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=[MeanAbsolutePercentageError()]
        )

        return model

    def set_data(self, target_data: Feature) -> "BaseForecastService":
        df = pd.DataFrame({'date': target_data.dates, 'goal': target_data.values})        

        return BaseForecastService(
            hparams=self._hparams, model=self._model, df=df
        ) 
                
    def preprocess_features(self) -> "BaseForecastService":
        """Предобработка признаков. Создадим переменные с лагом"""
        
        
        self._df.date = pd.to_datetime(self._df.date, format='%d.%m.%Y').dt.date
        self.df = self._df[self._df.date > date(year=2015, month=1, day=1)].sort_values(by='date', ascending=False)
        
        scaler = StandardScaler().fit(self._df[['goal']])
        self._df['z_goal'] = scaler.transform(self._df[['goal']])
        
        return BaseForecastService(hparams=self._hparams, model=self._model, df=self._df, scaler=scaler)
    
    def _prepare_rnn_data(self, data: Iterable) -> tuple[np.array, np.array]:
        """
        Создает батчи по lookback месяцев для иксов и по horizon месяца для игреков
        
        Также приводит к формату данных для RNN
        """
        
        data_range = range(self._hparams.lookback, len(data) - self._hparams.forecast_horizon + 1)
        
        x = np.array([data[i - self._hparams.lookback:i] for i in data_range])
        x = np.reshape(x, (x.shape[0], self._hparams.lookback, 1))

        y = np.array([data[i:i + self._hparams.forecast_horizon] for i in data_range])
        
        return x, y

    @lru_cache(maxsize=8)
    def _train(self, model, batch_size, epochs):
        X, y = self._prepare_rnn_data(self._df['z_goal'])
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        x_val, y_val = x_test[-12:], y_test[-12:]
        
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            epochs=epochs    
        )
        
        return model
    
    def train(self) -> "BaseForecastService":
        
        # TypeError: unhashable type: 'numpy.ndarray'
        model = self._train(self._model, self._hparams.batch_size, self._hparams.epochs)
        
        return BaseForecastService(hparams=self._hparams, model=model, df=self._df, scaler=self._scaler)
    
    @staticmethod
    def _score(model, x, y) -> ModelScore:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        
        y_pred = model.predict(x_test)

        return ModelScore(
            mape = mean_absolute_percentage_error(y_test, y_pred),
            r2_score = r2_score(y_test, y_pred)
        )
    
    @lru_cache(maxsize=8)
    def _predict(self, model):
        x, y = self._prepare_rnn_data(data=self._df['z_goal'])

        return model.predict(x[-1])
        
    
    def predict(self) -> ForecastResponse:
        # Берем самые последние данные, получаем предсказание и score для каждой модели
        y_pred = self._scaler.inverse_transform(self._predict(self._model))
        
        print(y_pred)
        
        x, y = self._prepare_rnn_data(data=self._df['z_goal'])
        
        return ForecastResponse(
            month_1=y_pred[0][0],
            month_2=y_pred[0][1],
            month_3=y_pred[0][2],
            scores=[]
        )
