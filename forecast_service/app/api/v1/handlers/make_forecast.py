from fastapi import APIRouter, HTTPException

from app.service.forecast_models import IPCForecast, IPPForecast, BaseForecastService

from app.schemas import (
    IndexFeaturesMapper,
    ForecastResponse,
    FeaturesResponse,
    IPPRequestCB,
    BaseRequest,
    ReadyOnModels
)
from app.schemas.io.request import IPCRequestCB
from app.schemas import IPPFeaturesResponse, IPCFeaturesResponse


forecast_router = APIRouter()


# @forecast_router.get("/{index}/features_list")
# async def features_list(index: ReadyOnModels):
#     """# Получить список названий всех признаков необходимых для обучения модели"""
#
#     try:
#         return IndexFeaturesMapper[index]()
#     except KeyError:
#         raise HTTPException(status_code=404, detail='Такого индекса нет')

@forecast_router.get("/ipp/features_list")
async def features_list():
    """# Получить список названий всех признаков необходимых для обучения модели"""

    return IPPFeaturesResponse()


@forecast_router.get("/ipc/features_list")
async def features_list():
    """# Получить список названий всех признаков необходимых для обучения модели"""

    return IPCFeaturesResponse()


@forecast_router.post("/base")
async def base_forecast(request: BaseRequest) -> ForecastResponse:
    """# Базовая модель для прогноза"""

    return (BaseForecastService(request.hparams)
            .set_data(request.target)
            .preprocess_features()
            .train()
            .predict())


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

    return (IPPForecast(request.hparams)
            .set_data(
                ipp=request.ipp,
                rzd=request.features.rzd,
                news=request.features.news,
                curs=request.features.curs,
                cb_monitor=request.features.cb_monitor,
                interest_rate=request.features.interest_rate,
                business_clim=request.features.business_clim,
                consumer_price=request.features.consumer_price
            )
            .preprocess_features()
            .train()
            .predict())


@forecast_router.post("/ipc/catboost")
async def cb_ipc_forecast(request: IPCRequestCB) -> ForecastResponse:
    """
    # Прогнозирование индекса потребительских цен с CatBoost

    ## Параметры:
    - __hparams:__ гиперпараметры CatBoost
    - __ipc:__ временной ряд индекса ИПП
    - __features:__ список временных рядов признаков индекса

    _Подробнее в описании DTO внизу страницы_

    ## Возвращает:
    ForecastResponse
    """

    return (IPCForecast(request.hparams)
            .set_data(
                ipc=request.ipc,
                curs=request.features.curs,
                interest_rate=request.features.interest_rate,
                money_supply=request.features.money_supply,
                agg_m0=request.features.agg_m0
            )
            .preprocess_features()
            .train()
            .predict())
