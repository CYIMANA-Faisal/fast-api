from http import HTTPStatus
from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from dto.create_sheep_dto import SheepDTO
from models.sheep_model import Sheep

from repositories import sheep_repository
from utils.api_response import format_response

router = APIRouter(
    prefix="/sheeps",
    tags=["sheeps"],
)


@router.post("/", response_model=List[Sheep])
async def add_sheep(sheep: SheepDTO):
    sheep = await sheep_repository.create(Sheep(sheep))
    response_data = format_response(
        HTTPStatus.OK, "Sheep created successfully", sheep)
    return JSONResponse(content=response_data)


@router.get("/", response_model=List[Sheep])
async def get_sheeps():
    sheeps = await sheep_repository.findAll()
    response_data = format_response(
        HTTPStatus.OK, "Sheeps retrieved successfully", sheeps)
    return JSONResponse(content=response_data)


@router.get("/{id}/", response_model=Sheep)
async def get_sheep(id: str):
    print(id)
    sheep = await sheep_repository.findById(int(id))
    if sheep is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="not found")
    response_data = format_response(
        HTTPStatus.OK, "Sheep retrieved successfully", sheep)
    return JSONResponse(content=response_data)
