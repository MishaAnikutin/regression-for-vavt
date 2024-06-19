from fastapi import APIRouter, HTTPException
from pipe import select
from whatever import _ 

from app.domain.forecast_models.ipp import catboost, rnn


from app.api.v1.schemas import (
    ForecastResponse,
    IPPRequestCB,
    IPPRequestRNN,
    IPPFeatures,
    FeatureResponse
)


forecast_router = APIRouter()


@forecast_router.get("/ipp/features_list")
async def features_list() -> FeatureResponse:
    """
    # Список признаков для индекса
    
    Получить список названий всех признаков необходимых для обучения модели
    
    _Подробнее в описании DTO внизу страницы_
    """
    
    fields = IPPFeatures.model_fields
    
    return FeatureResponse(feature_names=dict(zip(fields.keys(), fields.values() | select(_.description))))


@forecast_router.post("/ipp/catboost")
async def cb_ipp_forecast(request: IPPRequestCB) -> ForecastResponse:
    """
    # Прогнозирование индекса промышленного производства с CatBoost
    
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
        
    model = catboost.IPPForecast(dict(request.hparams))
    features = request.features
        
    values = [*dict(features).values(), list(request.ipp)]
    
    if any(map(lambda f: len(f) < 6, values)):
        raise HTTPException(status_code=400, detail="Неверные данные. Модель сдвигает данные на лаг до 6 месяцев. Если данных меньше, модель невозможно обучить")
    
    if len(set(map(len, values))) != 1:    
        raise HTTPException(status_code=400, detail="Неверные данные. Данные должны быть одной длины")
    
    model.set_data(
        goal=request.ipp, news=features.news, consumer_price=features.consumer_price,
        cb_monitor=features.cb_monitor, exchange_rate=features.interest_rate, 
        bussines_clim=features.bussines_clim, curs=features.curs, rzd=features.rzd 
    )
    
    try:
        model.train()
        forecast = model.predict()
    
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ошибка обучения модели: {exc}")
    forecast_response = ForecastResponse(month_1=forecast[0], month_2=forecast[1], month_3=forecast[2])
    
    return forecast_response


# @forecast_router.post("/ipp/rnn")
# async def rnn_ipp_forecast(request: IPPRequestRNN) -> ForecastResponse:
#     """
#     # БУДЕТ ГОТОВ В БУДУЩЕМ

#     # Прогнозирование индекса промышленного производства рекурентной нейронной сетью
    
#     ## Параметры:
#     - __hparams:__ гиперпараметры RNN
#     - __confidence_interval:__ доверительный интервал
#     - __goal:__ временной ряд индекса ИПП
#     - __features:__ список временных рядов признаков индекса
    
#     _Подробнее в описании DTO внизу страницы_

#     ## Возвращает:
#     ForecastResponse(month_1, month_2, month_3)
    
#     _т.е. прогноз на 1, 2 и 3 месяца соответственно_
#     """
    
#     # Not implemented
#     raise HTTPException(status_code=501)
