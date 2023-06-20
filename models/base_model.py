from typing import Optional
from datetime import datetime
from sqlmodel import Field, SQLModel


class BaseModel(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: Optional[datetime] = Field(default=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=datetime.utcnow)
