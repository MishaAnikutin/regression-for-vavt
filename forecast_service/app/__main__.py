import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import router


"""Запуск API Для прогнозного сервиса"""


app = FastAPI(title="iep-forecast-service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

uvicorn.run(app, host='0.0.0.0', port=5051)
