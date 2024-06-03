from backend.forecast.models.base import BaseForecast 


class ICPForecast(BaseForecast):

    feature_lags = {
    
    }
    
    def __init__(self, data, **kwargs) -> None:
        ... 