from fastapi import APIRouter
from .handlers import forecast_router


router_v1 = APIRouter()
router_v1.include_router(forecast_router)

__all__ = ['router_v1']