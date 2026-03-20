import logging


def history_to_sdapi_response(history_entry: dict, params: dict) -> dict:
    """
    Convert a ComfyUI history entry into an SDAPI-compatible response dict.

    Returns: {"images": [base64_str, ...], "parameters": {...}, "info": str}
    """
    images_b64 = []
    outputs = history_entry.get("outputs", {})

    for node_output in outputs.values():
        if "images_b64" in node_output:
            images_b64.extend(node_output["images_b64"])
        else:
            logging.warning("[sdapi] Node output has no images_b64: %s", node_output)

    status = history_entry.get("status") or {}
    success = status.get("completed", False)

    return {
        "images": images_b64,
        "parameters": params,
        "info": f"ComfyUI sdapi. success={success}",
    }
