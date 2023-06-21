from typing import Optional
from pydantic import BaseModel


class SheepDTO(BaseModel):
    tag: str
    farm_id: Optional[int]
