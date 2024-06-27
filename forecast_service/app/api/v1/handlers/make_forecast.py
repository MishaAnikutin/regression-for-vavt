from fastapi import APIRouter, HTTPException
from whatever import _ 

from app.domain.forecast_models.ipp import catboost, rnn


from app.api.v1.schemas import (
    ForecastResponse,
    IPPRequestCB,
    IPPRequestRNN,
    IPPFeatures,
    FeatureResponse,
    Feature
)


forecast_router = APIRouter()


@forecast_router.get("/ipp/features_list")
async def features_list() -> FeatureResponse:
    """
    # Список признаков для индекса
    
    Получить список названий всех признаков необходимых для обучения модели
    
    _Подробнее в описании DTO внизу страницы_
    """
            
    return FeatureResponse(
        features = {
            "news":           Feature(description="Новостной идекс ЦБ, Россия", dataset_uuid="423f7092-d29b-43da-8209-f100c1fc88cd"),
            "cb_monitor":     Feature(description="Промышленность. Как изменился спрос на продукцию, товары, услуги?", dataset_uuid="7d38c0d5-6bae-4856-b92c-9858223cfa89"),
            "bussines_clim":  Feature(description="Индикатор бизнес-климата ЦБ (промышленность), пункты, Россия", dataset_uuid="14c74eba-c1a7-4aff-a1e3-0aa473ce8062"),
            "interest_rate":  Feature(description="Базовая ставка - краткосрочная, %, Россия", dataset_uuid="87d93650-33f7-44b2-96df-6d520fa76c12"),
            "rzd":            Feature(description="Погрузка на сети РЖД, млн тонн, Россия", dataset_uuid="61265b19-7635-449c-9538-b45e078fb5e9"),
            "consumer_price": Feature(description="Индекс цен на электроэнергию в первой ценовой зоне, рублей за МВт/ч, Россия", dataset_uuid="f0873ac7-c29f-481b-8f47-e11c7b1b7c82"),
            "curs":           Feature(description="Официальный курс доллара США на заданную дату, устанавливаемый ежедневно", dataset_uuid="5e5ac82a-ab76-4567-b095-92f8064acb51")
        }
    )


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
        raise HTTPException(status_code=400, detail="Неверные данные. Данных меньше, чем на 6 месяцев, модель невозможно обучить")
    
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
