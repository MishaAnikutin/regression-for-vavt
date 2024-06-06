from fastapi import APIRouter
from .handlers import forecast_router


router = APIRouter()
router.include_router(forecast_router)

__all__ = ['router']