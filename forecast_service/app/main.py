from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.api.v1 import router_v1

from . import config

"""Запуск API Для прогнозного сервиса"""

forecast_service = FastAPI(
    title        = "iep-forecast-service",
    description  = "Сервис для прогноза макроиндексов",
    version      = "1.0",
    root_path    = config.APP_NGINX_PREFIX
)


forecast_service.add_middleware(
    CORSMiddleware,
    allow_origins     = config.APP_CORS_ORIGINS_LIST,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

forecast_service.include_router(router_v1, prefix='/v1')

forecast_service.openapi_schema = get_openapi(
    title   = "iep-forecast-service",
    version = "1.0",
    routes  = forecast_service.routes,
    servers = [{'url': config.APP_NGINX_PREFIX}]
)

# uvicorn app.main:forecast_service --reload --host 0.0.0.0 --port 5051
