import pandas as pd
import requests
import json
from pprint import pprint
from dataclasses import dataclass

"""
Пример работы с сервисом

Для подробной информации читайте README.md в этой папке
"""

# -------------------------------------------------------------------------------------
# Заглушки для имитации слоя с БД
 
@dataclass
class TimeSeriesDTO:
    """Некий DTO для общения с БД"""
    values: list[float]
    dates: list[str]


class DataLake:
    """Подобие слоя для работы с озером даных"""
    
    @staticmethod
    def get_data_by_id(dataset_uuid: str) -> TimeSeriesDTO:
        """
        подобие функции, которая по id возвращает временной ряд признака из озера данных
        
        просто открывает csv по ключу
        """
        
        tmp = pd.read_csv(f'./raw_data/{dataset_uuid}.csv', sep=';')
        
        return TimeSeriesDTO(
            values=list(tmp['values']),
            dates=list(tmp['date'])
        )

# -------------------------------------------------------------------------------------
# 1. Для индекса, который хотите предсказать, нужно получить список требуемых признаков

feature_list = requests.get(
    url='http://localhost:5051/forecast/api/v1/ipp/features_list'
).json()['features']

print("Требуемые признаки индекса промышленного производства:\n\n")
pprint(feature_list)
print()


# получаем месячные данные индекса пром. производства в % к соответствующему периоду предыдущего года 
ipp_data: TimeSeriesDTO = DataLake.get_data_by_id(
    dataset_uuid="c1c92863-1827-405e-b3e4-dea782f57316"
)

# сохраняем данные в нужном для запроса формате
ipp = {
    "dates": ipp_data.dates,
    "values": ipp_data.values
}

# -------------------------------------------------------------------------------------
# 2. Далее, получаем из озера данных данные каждого индекса по dataset_uuid
features = dict()

for feature in feature_list.keys():
    try:
        dataset_uuid = feature_list[feature]['dataset_uuid']
    except KeyError:
        print(f'{feature =}, {feature_list[feature]=}')

    data: TimeSeriesDTO = DataLake.get_data_by_id(dataset_uuid)
    
    features[feature] = {"dates": data.dates, "values": data.values}

print("Все полученные признаки:\n")

for key, value in features.items():
    print(f'{key}:\t{", ".join(map(str, map(lambda x: round(x, 2), value["values"][:5])))}...')

# -------------------------------------------------------------------------------------
# 3. После этого можно сформировать запрос для предсказания. 
# Получаем гиперпараметры модели и доверительный интервал

catboost_params = {"depth": 3, "learning_rate": 0.1, "l2_leaf_reg": 0.005, "iterations": 8}
confidence_interval = "90%"

# -------------------------------------------------------------------------------------
# 4. Формируем запрос
url = 'http://localhost:5051/forecast/api/v1/ipp/catboost'

data = {
    "hparams":             catboost_params,     # Гиперпараметры модели
    "confidence_interval": confidence_interval, # Доверительный интервал
    "ipp":                 ipp,                 # Данные ИПП 
    "features":            features             # Данные признаков
}

with open('data.json', 'w') as f:
    json.dump(data, f)

# -------------------------------------------------------------------------------------
# 5. Отправляем запрос
forecast_response = requests.post(url=url, json=data)

print()
print(f'status code: {forecast_response.status_code}')
# status code: 200

print('\nПрогноз:')
print(forecast_response.json()) 
# {'month_1': 103.71714137759228, 'month_2': 103.62474502878443, 'month_3': 104.02961034343352}
