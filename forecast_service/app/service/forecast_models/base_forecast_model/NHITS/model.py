import torch
import pandas as pd
from datetime import date

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import NHITS
from utilsforecast.losses import mape, rmse
from utilsforecast.evaluation import evaluate

from app.domain.forecast_interface import BaseForecast
from app.schemas import Feature, ForecastResponse, ModelScore
from app.schemas.ml.params import NHiTSHyperparameters


class BaseForecastService(BaseForecast):
    """Сервис для прогноза временных рядов с помощью рекурентных нейронных сетей"""

    last_day = date(year=2015, month=1, day=1)

    def __init__(
            self,
            hparams: NHiTSHyperparameters
    ):
        self._df = None
        self._hparams = hparams
        self._model = self._init_model()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _init_model(self):
        model = NeuralForecast(models=[
            NHITS(
                h=self._hparams.horizon,
                input_size=self._hparams.lookback,
                max_steps=self._hparams.epochs,
                scaler_type='standard',
                loss=MSE(),
                learning_rate=self._hparams.learning_rate,
                n_freq_downsample=[2, 1, 1],
                mlp_units=3 * [[2, 2]]
            )
        ], freq='M')

        # model.to(self.device)

        return model

    def set_data(self, target_data: Feature) -> "BaseForecastService":
        shape = len(target_data.dates)
        self._df = pd.DataFrame({'unique_id': ['1'] * shape, 'ds': target_data.dates, 'y': target_data.values})

        return self

    def preprocess_features(self) -> "BaseForecastService":
        # сделаем даты из строк в datetime.date
        self._df['ds'] = pd.to_datetime(self._df.ds, format='%d.%m.%Y')

        self._df = self._df.sort_values(by='ds')
        # Скалирование производит NeuralForecast под капотом
        return self

    def train(self) -> "BaseForecastService":
        if torch.cuda.is_available():
            self._df['y'] = self._df['y'].to(self.device)

        # размер валидационной выборки 12
        self._model.fit(df=self._df, val_size=12)

        return self

    def _score(self) -> ModelScore:
        Y_hat_insample = self._model.predict_insample(step_size=self._hparams.horizon).reset_index()

        evaluation_df = evaluate(Y_hat_insample.loc[:, Y_hat_insample.columns != 'cutoff'], metrics=[rmse, mape])

        return ModelScore(
            mape=evaluation_df[evaluation_df['metric'] == 'mape']['NHITS'].iloc[0],
            r2_score=1
        )

    def predict(self) -> ForecastResponse:
        # Предсказание модели предыдущих значений
        previous = list(self._model.predict_insample(step_size=self._hparams.horizon).reset_index().y)

        # Предсказание будущих значений
        predict = list(self._model.predict(self._df).reset_index()['NHITS'])

        # Получаем score
        scores = [self._score()]

        return ForecastResponse(
            previous=previous,
            predict=predict,
            scores=scores
        )
