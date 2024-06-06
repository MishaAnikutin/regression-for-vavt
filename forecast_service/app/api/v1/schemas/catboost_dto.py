from pydantic import BaseModel


class CatBoostHyperparameters(BaseModel):
    depth:         int  
    learning_rate: float 
    l2_leaf_reg:   float 
    iterations:    int 
