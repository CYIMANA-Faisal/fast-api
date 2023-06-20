from typing import Optional
from models.base_model import BaseModel
from sqlmodel import Field


class User(BaseModel, table=True):
    name: str
    email: str
    password: str
