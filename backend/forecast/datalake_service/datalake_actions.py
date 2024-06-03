import pandas as pd 
import numpy as np


# Тут для примера, я просто из excel беру данные
def get_index_data(session, index_name):
    df = pd.read_csv("../../../data/preprocessed_data.csv")
    
    return df[index_name]

