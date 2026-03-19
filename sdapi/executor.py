import asyncio
import time
import uuid

import execution

SDAPI_TIMEOUT = 300.0

_server = None


def set_server(server):
    global _server
    _server = server


async def submit_and_wait(workflow: dict) -> dict:
    """
    Submit a ComfyUI workflow and wait synchronously for completion.
    Returns the history entry dict. Raises RuntimeError or asyncio.TimeoutError.
    """
    server = _server
    if server is None:
        raise RuntimeError("Server not initialized")

    prompt_id = str(uuid.uuid4())

    valid = await execution.validate_prompt(prompt_id, workflow, None)
    if not valid[0]:
        raise RuntimeError(f"Workflow validation failed: {valid[1]}")

    outputs_to_execute = valid[2]

    event = asyncio.Event()
    server.pending_sdapi[prompt_id] = event

    try:
        number = server.number
        server.number += 1
        extra_data = {"create_time": int(time.time() * 1000)}

        server.prompt_queue.put(
            (number, prompt_id, workflow, extra_data, outputs_to_execute, {})
        )

        await asyncio.wait_for(event.wait(), timeout=SDAPI_TIMEOUT)

        history = server.prompt_queue.get_history(prompt_id=prompt_id)
        return history.get(prompt_id, {})

    finally:
        server.pending_sdapi.pop(prompt_id, None)
