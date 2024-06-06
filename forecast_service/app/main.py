from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import router


"""Запуск API Для прогнозного сервиса"""


forecast_service = FastAPI(title="iep-forecast-service")

forecast_service.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

forecast_service.include_router(router)
