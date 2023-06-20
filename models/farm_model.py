from sqlmodel import Field
from models.base_model import BaseModel


class Farm(BaseModel, table=True):
    name: str = Field(index=True)
    location: str
