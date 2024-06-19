from pydantic import BaseModel, Field


class CatBoostHyperparameters(BaseModel):
    depth:         int   = Field(default=3)
    learning_rate: float = Field(default=0.1)
    l2_leaf_reg:   float = Field(default=.005)
    iterations:    int   = Field(default=8)
