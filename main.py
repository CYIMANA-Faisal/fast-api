from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlmodel import SQLModel, create_engine
from http import HTTPStatus
from utils.api_response import format_response
from models import farm_model, sheep_model, user_model, prediction_model

app = FastAPI()

engine = create_engine("sqlite:///database.db", echo=True)
SQLModel.metadata.create_all(engine)


@app.get("/")
def root():
    response_data = format_response(
        HTTPStatus.OK, "welcome to Cool sheep AI API!", None)
    return JSONResponse(content=response_data)
