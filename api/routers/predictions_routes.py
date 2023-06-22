from http import HTTPStatus
from typing import Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from model.main import get_value
import os
from utils.api_response import format_response

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
)


@router.post("/")
async def upload_images(file: Optional[UploadFile] = None):
    # try:
    #     current_dir = os.getcwd()
    #     relative_path = "imgs"
    #     folder_path = os.path.join(current_dir, relative_path)
    #     file_names = os.listdir(folder_path)
    #     results = await get_value(f'{folder_path}/{file_names[0]}')
    #     response_data = format_response(
    #         HTTPStatus.CREATED, "Files are being processed by the model!", {"image_filenames": results})
    #     return JSONResponse(content=response_data)
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="An error occured when uploading the files")
    current_dir = os.getcwd()
    relative_path = "imgs"
    folder_path = os.path.join(current_dir, relative_path)
    file_names = os.listdir(folder_path)
    results = get_value(f'{folder_path}/{file_names[0]}')
    response_data = format_response(
        HTTPStatus.CREATED, "Files are being processed by the model!", {"image_filenames": results})
    return JSONResponse(content=response_data)
