"""
SDAPI FastAPI application.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sdapi.routes import router

app = FastAPI(title="ComfyUI SDAPI", version="1.0.0")
app.include_router(router, prefix="/sdapi/v1")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse({"error": "Internal server error"}, status_code=500)
