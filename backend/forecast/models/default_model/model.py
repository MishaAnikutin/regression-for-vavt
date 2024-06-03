from backend.forecast.models.base import BaseForecast 


class DefaultForecast(BaseForecast):
    def __init__(self, data) -> None:
        ... 