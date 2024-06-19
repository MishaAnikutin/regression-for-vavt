import os 


APP_CORS_ORIGINS_LIST = os.getenv(
    'APP_CORS_ORIGINS_LIST', 
    default="https://data.vavt.ru,http://data.vavt.ru,http://localhost:3000"
).split(',')


APP_NGINX_PREFIX = os.getenv(
    'APP_NGINX_PREFIX',
    default="/forecast/api"    
)