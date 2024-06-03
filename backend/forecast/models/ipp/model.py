from backend.forecast.models.base import BaseForecast 


class IPPForecast(BaseForecast):
    """
    Модель прогноза Индекса промышленного производства на CatBoost
    
    При предобработке данных, разные признаки мы будем сдвигать по определенному лагу
    
    Условно, шоки поставок РЖД повлияют на экономику спустя только несколько
    месяцев
    
    Поэтому, будем хранить словарь с маппингом столбцов и их лаггом 
    """
    
    feature_lags = {
        "news": 1,           # Новостной индекс ЦБ
        "consumer_price": 0, # 
        "fin_rez_lag": 2,    #
        "bussines_clim": 2,  #
        "curs": 0,           #
        "rzd": 1             #
    }
    
    def __init__(self, **hparams) -> None:
        self.__hparams = hparams
        
    
    def _preprocess_data(self):
        ...
        
    