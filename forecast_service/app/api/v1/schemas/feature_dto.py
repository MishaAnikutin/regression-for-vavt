from pydantic import BaseModel, Field


class FeatureRequest(BaseModel):
    index_name: str = Field(
        default="Индекс промышленного производства",
        description="Название индекса (временного ряда) для которого хотите получить список признаков для модели"
    )

class FeatureResponse(BaseModel):
    """Модель получения всех признаков для модели. Ответом будет словарь в формате признак: описание"""
    feature_names: dict[str, str]
