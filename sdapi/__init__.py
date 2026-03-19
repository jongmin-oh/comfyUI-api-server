"""
SDAPI FastAPI application.
"""
from fastapi import FastAPI
from sdapi.routes import router

app = FastAPI(title="ComfyUI SDAPI", version="1.0.0")
app.include_router(router, prefix="/sdapi/v1")
