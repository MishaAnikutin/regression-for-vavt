from fastapi import APIRouter, HTTPException

from app.api.v1.schemas import (ForecastResponse, IPPRequest, FeatureRequest, FeatureResponse)
from app.domain.forecast_models import IPPForecast
from app.api.v1.mapping import FeatureMapper

forecast_router = APIRouter()


@forecast_router.post("/features_list")
async def features_list(request: FeatureRequest) -> FeatureResponse:
    """
    # Список признаков для индекса
    
    Получить список названий всех признаков необходимых для обучения модели
    
    ## Параметры:
    -   __index_name:__ Название индекса (временного ряда) для которого хотите получить список признаков для модели
    
    ## Возвращает:
    - __feature_names:__ Словарь имя-описание признаков. 
    - __Status Code 422,__ если название индекса не было найдено (значит, для него нужно будет использовать стандартную модель)  
    _Подробнее в описании DTO внизу страницы_
    """
     
    index_name = request.index_name
    try:
        return FeatureResponse(feature_names=FeatureMapper[index_name])
    except KeyError:
        raise HTTPException(status_code=422, detail='Для этого индекса еще не определ список признаков')        


@forecast_router.post("/ipp")
async def ipp_forecast(request: IPPRequest) -> ForecastResponse:
    """
    # Прогнозирование индекса промышленного производства
    
    ## Параметры:
    - __hparams:__ гиперпараметры CatBoost
    - __confidence_interval:__ доверительный интервал
    - __goal:__ временной ряд индекса ИПП
    - __features:__ список временных рядов признаков индекса
    
    _Подробнее в описании DTO внизу страницы_

    ## Возвращает:
    ForecastResponse(month_1, month_2, month_3)
    
    _т.е. прогноз на 1, 2 и 3 месяца соответственно_
    """
        
    model = IPPForecast(dict(request.hparams))
    features = request.features
    
    model.set_data(
        goal=request.ipp, news=features.news, consumer_price=features.consumer_price,
        cb_monitor=features.cb_monitor, exchange_rate=features.exchange_rate, 
        bussines_clim=features.bussines_clim, curs=features.curs, rzd=features.rzd 
    )
    
    model.train()
    forecast = model.predict()
    
    forecast_response = ForecastResponse(month_1=forecast[0], month_2=forecast[1], month_3=forecast[2])
    
    return forecast_response