from fastapi import APIRouter, HTTPException

from app.api.v1.schemas import (ForecastResponse, IPPRequest, IPPFeatures, FeatureRequest, FeatureResponse)

from app.domain.forecast_models import IPPForecast
from app.api.v1.mapping import FeatureMapper

forecast_router = APIRouter()


@forecast_router.post("/ipp/features_list")
async def features_list() -> FeatureResponse:
    """
    # Список признаков для индекса
    
    Получить список названий всех признаков необходимых для обучения модели
    
    _Подробнее в описании DTO внизу страницы_
    """

    ipp_feature_keys = IPPFeatures.model_fields.keys()
    
    ipp_feature_desctiprion = list(
        map(
            lambda feature: feature.description,
            IPPFeatures.model_fields.values()
        )
    )
    
    ipp_features = {
        key: description
        for key, description in
        zip(ipp_feature_keys, ipp_feature_desctiprion)
    }

    return FeatureResponse(feature_names=ipp_features)


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