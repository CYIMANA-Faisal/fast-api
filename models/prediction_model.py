from typing import Optional
from models.base_model import BaseModel
from sqlmodel import Field, Relationship


class Prediction(BaseModel, table=True):
    predicted_weight: float = Field(index=True)
    actual_weight: float = Field(index=True)
    sheep_id: Optional[int] = Field(default=None, foreign_key="sheep.id")
    sheep: Optional["sheep_model.Sheep"] = Relationship(
        back_populates="predictions")
