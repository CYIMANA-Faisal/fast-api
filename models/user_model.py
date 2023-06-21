from pydantic import EmailStr
from sqlmodel import Field
from models.base_model import BaseModel


class User(BaseModel, table=True):
    name: str
    email: EmailStr
    password: str = Field(max_length=256, min_length=20)
