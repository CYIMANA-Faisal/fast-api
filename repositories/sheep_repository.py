from typing import List, Optional
from sqlmodel import select
from models.sheep_model import Sheep
from db.engine import session


async def findAll() -> List[Sheep]:
    with session:
        statement = select(Sheep)
        result = session.exec(statement)
        return result.all()


async def findById(id: int) -> Optional[Sheep]:
    with session:
        statement = select(Sheep).where(Sheep.id == id)
        result = session.exec(statement)
        return result.one_or_none()


async def create(sheep: Sheep) -> Optional[Sheep]:
    with session:
        session.add(sheep)
        return session.commit()
