from fastapi import FastAPI
from fastapi.responses import JSONResponse
from http import HTTPStatus
from utils.api_response import format_response
from api.routers import sheep_routes

app = FastAPI()
app.include_router(sheep_routes.router)


@app.get("/")
def root():
    response_data = format_response(
        HTTPStatus.OK, "Welcome to cool sheep ai API!", None)
    return JSONResponse(content=response_data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
