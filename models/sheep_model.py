from typing import Optional
from sqlmodel import Field
from models.base_model import BaseModel


class Sheep(BaseModel, table=True):
    tag: str = Field(index=True)
    farm_id: Optional[int] = Field(default=None, foreign_key="farm.id")
