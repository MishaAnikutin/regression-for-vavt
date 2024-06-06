from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import router_v1


"""Запуск API Для прогнозного сервиса"""


forecast_service = FastAPI(
    title="iep-forecast-service",
    description="Сервис для прогноза макроиндексов",
    version="1.0",
    docs_url='/docs',
    openapi_url='/openapi.json',
    redoc_url=None
)

forecast_service.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

forecast_service.include_router(router_v1, prefix="/v1")
