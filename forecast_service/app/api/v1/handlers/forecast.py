from fastapi import APIRouter, HTTPException
from app.service.forecast_models.ipp import catboost, rnn


from app.schemas import (
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
    # Список признаков для прогноза индекса промышленного производства, Россия: ежемесячные данные в % к соответствующему периоду предыдущего года
    
    __dataset_uuid = c1c92863-1827-405e-b3e4-dea782f57316__
    
    Получить список названий всех признаков необходимых для обучения модели
    
    _Подробнее в описании DTO внизу страницы_
    """
            
    return FeatureResponse(
        features={
            "news":           Feature(description="Новостной индекс ЦБ, Россия", dataset_uuid="423f7092-d29b-43da-8209-f100c1fc88cd"),
            "cb_monitor":     Feature(description="Оценка изменения спроса на продукцию, товары, услуги (промышленность), пункты, Россия", dataset_uuid="7d38c0d5-6bae-4856-b92c-9858223cfa89"),
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
    ForecastResponse(responses: tuple[float])
    
    _т.е. прогноз на следующие 3 месяца_
    """
    
    result = catboost\
        .IPPForecast(dict(request.hparams))\
        .set_data(
            ipp            = request.ipp,
            rzd            = request.features.rzd,
            news           = request.features.news,
            curs           = request.features.curs,
            cb_monitor     = request.features.cb_monitor,
            interest_rate  = request.features.interest_rate,
            bussines_clim  = request.features.bussines_clim,
            consumer_price = request.features.consumer_price
        )\
        .preprocess_features()\
        .train()\
        .predict()
    
    return ForecastResponse(
        month_1 = result.month_1,
        month_2 = result.month_2,
        month_3 = result.month_3,
        scores  = result.scores
    )

