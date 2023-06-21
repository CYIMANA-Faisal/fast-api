from typing import List
from sqlmodel import Field, Relationship
from models.base_model import BaseModel


class Farm(BaseModel, table=True):
    name: str = Field(index=True)
    location: str
    sheep: List["sheep_model.Sheep"] = Relationship(back_populates="farm")
