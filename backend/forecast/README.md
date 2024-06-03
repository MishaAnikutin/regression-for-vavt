## Сервис для прогнозов макроиндексов

### Данные с фронтенда:
- Название индекса (желательно после слоя DTO, для унификации)

### Пример работы
1. Через фронтенд получаем название индекса который прогнозируем, а также его гиперпараметры

__Запрос в API предполагается по следующей схеме:__
```json
{
    "index_name": "Индекс потребительских цен",      // имя индекса
    "model_kwargs": {                                // гиперпараметры модели, например: 
        "": ,
        "": 
    }
}

```

2. По названию индекса получаем с озера данных значения индекса и значения необходимых для него признаков
3. Далее, по фабрике прогнозных моделей, получаем необходимую модель

__Все модели имплементируют следующий интерфейс:__

```python
class BaseForecast(ABC):
    """Интерфейс предсказания индекса"""
    
    @abstractmethod
    def __init__(self, **hparams) -> None:
        ...
    
    @property
    @abstractmethod
    def GetRequiredFeatures(self) -> list[str]:
        ... 
    
    @abstractmethod
    def set_data(self, index_data: np.ndarray, feature_data: np.ndarray[np.ndarray[float]]) -> None:
        ...
    
    @abstractmethod
    def train_model(self) -> None:
        ...
    
    @abstractmethod
    def predict(self) -> list[tuple[float]]:
        ...   
    
```

Вся реализация моделей инкапсулирована в его дочерних классах: предобработка данных, деление на обучающую и тестовую выборку, выбор модели машинного обучения и прочее. На слое сервисов, достаточно знать этих методов, которые повторяются в каждой модели. 

Благодаря такому подходу будет легко внедрять новые модели: для новой модели достаточно будет унаследовать этот класс и в фабрике добавить её в Mapper   

4. Устанавливаем данные, обучаем модель, по методу predict получаем результат

__Пример:__

```python
import numpy as np
from django.http import HttpResponse

from backend.forecast.model import BaseForecast
from backend.forecast.factory import ForecastFactory
from backend.forecast.datalake_service import get_index_data, get_features_data


def make_forecast(request, session) -> HttpResponse:
    # Получаем из запроса название индекса который прогнозируем, а также гиперпараметры модели
    index_name, model_kwargs = request.index_name, request.model_kwargs

    ForecastModel: BaseForecast = ForecastFactory.get_model(index_name)

    model = ForecastModel(model_kwargs)
    required_features: list[str] = ForecastModel.GetRequiredFeatures

    # Получаем данные самого индекса
    index_data: np.ndarray = get_index_data(index_name, session)

    # Получаем данные необходимых признаков для этого индекса
    features_data: np.ndarray[np.ndarray] = get_features_data(required_features, session) 

    model.set_data(index_data=index_data, features_data=features_data)
    model.train()
    forecast = model.predict()
    
    return HttpResponse({"forecast": forecast}, status=200)

```