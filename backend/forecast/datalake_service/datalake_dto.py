from dataclasses import dataclass 

class DataLakeDTO:
    """Родительский класс для всех DTO из озера данных"""
    
    @dataclass
    class Data:
        ...


class IPPDTO(DataLakeDTO):
    """DTO для индекса пром производства"""
    
    @dataclass 
    class Data:
        ipp: list[float]             # Индекс пром производства 
        news: list[float]            # Новостной индекс ЦБ
        cb_monitor: list[float]      # Промышленность. Как изменился спрос на продукцию, товары, услуги?
        bussines_clim: list[float]   # Промышленность Индикатор бизнес-климата Банка России
        exchange_rate: list[float]   # Ключевая ставка 
        rzd: list[float]             # Погрузка на сети РЖД
        consumer_price: list[float]  # Индекс цен на электроэнергию в первой ценовой зоне
        curs: list[float]            # Курс рубля к доллару США
        
        
        
