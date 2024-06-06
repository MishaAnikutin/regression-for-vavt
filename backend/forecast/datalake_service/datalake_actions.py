from .datalake_dto import DataLakeDTO, IPPDTO
import pandas as pd 


def get_datalake_data(db_session, index_name: str, required_features: list[str]) -> DataLakeDTO.Data:
    # Предполакается следующее:
    # 
    # with db_session as session:
    #    stmt = f"SELECT (?, {'?,' * len(required_features)}) FROM datalake"
    #    session.execute(stmt, index_name, *required_features)
    #    data = session.fetchall()
    # 
    # далее по фабрике для каждого признака получаем свой DTO и устанавливаем ему значения
    # Тут для примера беру именно для ИПП из таблички, вам виднее как это лучше организовать

    df = pd.read_csv("../../../data/preprocessed_data.csv")
    
    data = IPPDTO.Data(
        ipp=df['goal'],
        news=df['news'],
        cb_monitor=df['cb_monitor'],     
        bussines_clim=df['bussines_clim'],  
        exchange_rate=df['exchange_rate'],
        rzd=df['rzd'],            
        consumer_price=df['consumer_price'],
        curs=df['curs']
    )
    
    return data

