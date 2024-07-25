# from service.forecast_models.base_forecast_model.RNN.model import BaseForecastService
from app.service.forecast_models.base_forecast_model.NHITS.model import BaseForecastService


def get_base_model():
    return BaseForecastService


__all__ = [
    'get_base_model',
]
