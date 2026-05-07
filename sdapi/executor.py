import asyncio
import logging
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
        error = valid[1]
        error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise RuntimeError(f"Workflow validation failed: {error_msg}")

    outputs_to_execute = valid[2]

    event = asyncio.Event()
    server.pending_sdapi[prompt_id] = event

    queue_size = server.prompt_queue.get_tasks_remaining()
    if queue_size > 0:
        logging.warning("[sdapi] prompt_id=%s: queue has %d pending task(s) before submit", prompt_id, queue_size)

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

    except asyncio.TimeoutError:
        logging.error("[sdapi] prompt_id=%s timed out after %.0fs — removing from queue", prompt_id, SDAPI_TIMEOUT)
        raise

    finally:
        server.pending_sdapi.pop(prompt_id, None)
        # 큐에서 대기 중인 경우에만 제거 (이미 실행 중이면 delete_queue_item이 건드리지 않음)
        server.prompt_queue.delete_queue_item(lambda p: p[1] == prompt_id)
