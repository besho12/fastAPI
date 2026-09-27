import os

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(
    prefix="/api/v",
    tags=["Update"],
)



LATEST_VERSION = os.getenv("UPDATE_LATEST_VERSION", "0.1.0")
LATEST_BUILD = int(os.getenv("UPDATE_LATEST_BUILD", "1"))

DOWNLOAD_URL = os.getenv(
    "UPDATE_DOWNLOAD_URL",
    "https://2.24.128.5/downloads/RoleFit_Setup.exe",
)

FORCE_UPDATE = os.getenv("UPDATE_FORCE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}

RELEASE_NOTES = os.getenv(
    "UPDATE_RELEASE_NOTES",
    "تحسينات في سرعة المقارنة وإصلاح بعض الأخطاء.",
)


class UpdateInfo(BaseModel):
    latest_version: str
    latest_build: int
    download_url: str
    force_update: bool
    release_notes: str


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


@router.get(
    "/check-update",
    response_model=UpdateInfo,
    response_class=UTF8JSONResponse,
)
def check_update(response: Response) -> UpdateInfo:
    response.headers["Cache-Control"] = "no-store"

    return UpdateInfo(
        latest_version=LATEST_VERSION,
        latest_build=LATEST_BUILD,
        download_url=DOWNLOAD_URL,
        force_update=FORCE_UPDATE,
        release_notes=RELEASE_NOTES,
    )