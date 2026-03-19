import base64
import logging
import os

import folder_paths


def history_to_sdapi_response(history_entry: dict, params: dict) -> dict:
    """
    Convert a ComfyUI history entry into an SDAPI-compatible response dict.

    Returns: {"images": [base64_str, ...], "parameters": {...}, "info": str}
    """
    images_b64 = []
    outputs = history_entry.get("outputs", {})

    for node_output in outputs.values():
        if "images" not in node_output:
            continue
        for img_info in node_output["images"]:
            img_path = _resolve_image_path(
                img_info.get("filename"),
                img_info.get("subfolder", ""),
                img_info.get("type", "output"),
            )
            if img_path is None or not os.path.isfile(img_path):
                logging.warning("[sdapi] Image file not found: %s", img_path)
                continue
            with open(img_path, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode("utf-8"))

    status = history_entry.get("status") or {}
    success = status.get("completed", False)

    return {
        "images": images_b64,
        "parameters": params,
        "info": f"ComfyUI sdapi. success={success}",
    }


def _resolve_image_path(filename: str | None, subfolder: str, img_type: str) -> str | None:
    if not filename:
        return None
    if img_type == "output":
        base_dir = folder_paths.get_output_directory()
    elif img_type == "temp":
        base_dir = folder_paths.get_temp_directory()
    elif img_type == "input":
        base_dir = folder_paths.get_input_directory()
    else:
        return None
    if subfolder:
        base_dir = os.path.join(base_dir, subfolder)
    return os.path.join(base_dir, filename)
