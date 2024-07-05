from pydantic import BaseModel, Field, UUID1


class FeatureRequest(BaseModel):
    index_name: str = Field(
        default="Индекс промышленного производства",
        description="Название индекса (временного ряда) для которого хотите получить список признаков для модели"
    )

class Feature(BaseModel):
    dataset_uuid: str
    description: str

class FeatureResponse(BaseModel):
    """Модель получения всех признаков для модели. Ответом будет словарь в формате признак: описание"""
    features: dict[str, Feature]
