import os
from dotenv import load_dotenv


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)


APP_CORS_ORIGINS_LIST = os.getenv('APP_CORS_ORIGINS_LIST',
                                  default="https://data.vavt.ru,http://data.vavt.ru,http://localhost:3000"
                                  ).split(',')


APP_NGINX_PREFIX = os.getenv('APP_NGINX_PREFIX',
                             default="/forecast/api"
                             )

BASE_FORECAST_MODEL = os.getenv('BASE_FORECAST_MODEL')
