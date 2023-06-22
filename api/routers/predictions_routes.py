from http import HTTPStatus
from typing import Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil

from utils.api_response import format_response

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
)


@router.post("/")
async def upload_images(file: Optional[UploadFile] = None):
    try:
        response_data = format_response(
            HTTPStatus.CREATED, "Files are being processed by the model!", {"image_filenames": {"image_filenames": file.filename}})
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="An error occured when uploading the files")
