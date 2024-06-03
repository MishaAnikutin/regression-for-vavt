from backend.forecast.models import (
    BaseForecast,
    ICPForecast,
    IPPForecast,
    DefaultForecast
)


class ForecastFactory:
    # TODO: Заменить строки на обьекты
         
    ForecastMapping = {
        "Индекс потребительских цен": ICPForecast,
        "Индекс промышленного производства": IPPForecast
    }
   
    @classmethod
    def get_model(cls, index_name) -> BaseForecast:
        return cls.ForecastMapping.get(index_name, default=DefaultForecast)
