from typing import TypeAlias, Union

from pydantic import BaseModel, Field
from app.schemas.ml.scores import ModelScore


class ForecastResponse(BaseModel):
    """Прогноз на предыдущие значения, на будущие и качество модели"""

    previous: list[float]
    predict: list[float]
    scores: list[ModelScore]


class FeatureResponse(BaseModel):
    dataset_uuid: str
    description: str


class IPPFeaturesResponse(BaseModel):
        news:           FeatureResponse = FeatureResponse(description="Новостной индекс ЦБ, Россия", dataset_uuid="423f7092-d29b-43da-8209-f100c1fc88cd")
        cb_monitor:     FeatureResponse = FeatureResponse(description="Оценка изменения спроса на продукцию, товары, услуги (промышленность), пункты, Россия", dataset_uuid="7d38c0d5-6bae-4856-b92c-9858223cfa89")
        business_clim:  FeatureResponse = FeatureResponse(description="Индикатор бизнес-климата ЦБ (промышленность), пункты, Россия", dataset_uuid="14c74eba-c1a7-4aff-a1e3-0aa473ce8062")
        interest_rate:  FeatureResponse = FeatureResponse(description="Базовая ставка - краткосрочная, %, Россия", dataset_uuid="87d93650-33f7-44b2-96df-6d520fa76c12")
        rzd:            FeatureResponse = FeatureResponse(description="Погрузка на сети РЖД, млн тонн, Россия", dataset_uuid="61265b19-7635-449c-9538-b45e078fb5e9")
        consumer_price: FeatureResponse = FeatureResponse(description="Индекс цен на электроэнергию в первой ценовой зоне, рублей за МВт/ч, Россия", dataset_uuid="f0873ac7-c29f-481b-8f47-e11c7b1b7c82")
        curs:           FeatureResponse = FeatureResponse(description="Официальный курс доллара США на заданную дату, устанавливаемый ежедневно", dataset_uuid="5e5ac82a-ab76-4567-b095-92f8064acb51")


class IPCFeaturesResponse(BaseModel):
        interest_rate:  FeatureResponse = FeatureResponse(description="Базовая ставка - краткосрочная, %, Россия", dataset_uuid="87d93650-33f7-44b2-96df-6d520fa76c12")
        curs:           FeatureResponse = FeatureResponse(description="Официальный курс доллара США на заданную дату, устанавливаемый ежедневно", dataset_uuid="5e5ac82a-ab76-4567-b095-92f8064acb51")
        money_supply:   FeatureResponse = FeatureResponse(description="Денежная масса, млрд нац ден ед, Россия", dataset_uuid='92b26933-97ab-45ab-9d52-18f509e193cb')
        m0_agg:         FeatureResponse = FeatureResponse(description="Денежный аггрегат М0, Россия", dataset_uuid='None')


FeaturesResponse: TypeAlias = Union[IPCFeaturesResponse, IPPFeaturesResponse]
