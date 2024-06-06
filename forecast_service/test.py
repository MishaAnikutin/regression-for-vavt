from pydantic import BaseModel, Field
    
class IPPFeatures(BaseModel):
    news:           list[float] = Field(None, description="Новостной индекс ЦБ")
    cb_monitor:     list[float] = Field(None, description="Промышленность. Как изменился спрос на продукцию, товары, услуги?") 
    bussines_clim:  list[float] = Field(None, description="Промышленность Индикатор бизнес-климата Банка России") 
    exchange_rate:  list[float] = Field(None, description="Ключевая ставка ")
    rzd:            list[float] = Field(None, description="Погрузка на сети РЖД")
    consumer_price: list[float] = Field(None, description="Индекс цен на электроэнергию в первой ценовой зоне")
    curs:           list[float] = Field(None, description="Курс рубля к доллару США")



print(IPPFeatures.model_fields.keys())
print(list(map(lambda feature: feature.description, IPPFeatures.model_fields.values())))
