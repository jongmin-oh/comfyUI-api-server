import asyncio
import logging
import traceback

import folder_paths
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from sdapi.executor import submit_and_wait
from sdapi.models import Img2ImgRequest, Txt2ImgRequest
from sdapi.serializer import history_to_sdapi_response
from sdapi.workflow_builder import (
    SDAPI_SAMPLER_MAP,
    build_img2img_workflow,
    build_txt2img_workflow,
)

router = APIRouter()


@router.post("/txt2img")
async def txt2img(body: Txt2ImgRequest) -> JSONResponse:
    params = body.model_dump()

    try:
        workflow, params["seed"] = build_txt2img_workflow(params)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception:
        logging.error("[sdapi] Workflow build error:\n%s", traceback.format_exc())
        return JSONResponse({"error": "Internal error building workflow"}, status_code=500)

    try:
        history_entry = await submit_and_wait(workflow)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "Generation timed out"}, status_code=504)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception:
        logging.error("[sdapi] Execution error:\n%s", traceback.format_exc())
        return JSONResponse({"error": "Internal error during generation"}, status_code=500)

    status = history_entry.get("status") or {}
    if status and not status.get("completed", True):
        return JSONResponse(
            {"error": "Generation failed", "details": status.get("messages", [])},
            status_code=500,
        )

    return JSONResponse(history_to_sdapi_response(history_entry, params))


@router.post("/img2img")
async def img2img(body: Img2ImgRequest) -> JSONResponse:
    params = body.model_dump()

    try:
        workflow, params["seed"] = build_img2img_workflow(params, body.init_images[0])
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception:
        logging.error("[sdapi] Workflow build error:\n%s", traceback.format_exc())
        return JSONResponse({"error": "Internal error building workflow"}, status_code=500)

    try:
        history_entry = await submit_and_wait(workflow)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "Generation timed out"}, status_code=504)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception:
        logging.error("[sdapi] Execution error:\n%s", traceback.format_exc())
        return JSONResponse({"error": "Internal error during generation"}, status_code=500)

    status = history_entry.get("status") or {}
    if status and not status.get("completed", True):
        return JSONResponse(
            {"error": "Generation failed", "details": status.get("messages", [])},
            status_code=500,
        )

    return JSONResponse(history_to_sdapi_response(history_entry, params))


@router.get("/sd-models")
async def list_models() -> JSONResponse:
    models = folder_paths.get_filename_list("checkpoints")
    return JSONResponse([
        {"title": m, "model_name": m, "filename": m, "hash": "", "sha256": ""}
        for m in models
    ])


@router.get("/samplers")
async def list_samplers() -> JSONResponse:
    return JSONResponse([
        {"name": name, "aliases": [], "options": {}}
        for name in SDAPI_SAMPLER_MAP
    ])
