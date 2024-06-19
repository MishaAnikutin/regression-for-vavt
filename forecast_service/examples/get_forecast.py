import pandas as pd
import requests
from pprint import pprint

df = pd.read_csv('examples/preprocessed_ipp_factors.csv')


feature_list = requests.get(url='http://localhost:5051/forecast/api/v1/ipp/features_list').json()
pprint(feature_list)

catboost_params = {"depth": 3, "learning_rate": 0.1, "l2_leaf_reg": 0.005, "iterations": 8}


ipp = list(df.goal)
features = {
    "news": list(df.news),
    "cb_monitor": list(df.cb_monitor),
    "bussines_clim": list(df.bussines_clim),
    "interest_rate": list(df.interest_rate),
    "rzd": list(df.rzd),
    "consumer_price": list(df.consumer_price),
    "curs": list(df.curs)
}

forecast_response = requests.post(
    url  = 'http://localhost:5051/forecast/api/v1/ipp/catboost',
    json = {
        "hparams": catboost_params,
        "confidence_interval": "90%",
        "ipp": ipp,
        "features": features
    }
)

print(f'status code: {forecast_response.status_code}')
# status code: 200

print(forecast_response.json()) 
# {'month_1': 103.71714137759228, 'month_2': 103.62474502878443, 'month_3': 104.02961034343352}
