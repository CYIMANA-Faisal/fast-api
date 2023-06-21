from typing import List, Optional
from sqlmodel import Field, Relationship
from models.base_model import BaseModel
from models.farm_model import Farm
from models.prediction_model import Prediction


class Sheep(BaseModel, table=True):
    tag: str = Field(index=True)
    farm_id: Optional[int] = Field(default=None, foreign_key="farm.id")
    farm: Optional["Farm"] = Relationship(back_populates="sheep")
    predictions: List["Prediction"] = Relationship(
        back_populates="sheep")
