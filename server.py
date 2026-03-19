"""
Minimal PromptServer stub — replaces the aiohttp-based server.
Provides just enough interface for PromptExecutor and PromptQueue.
"""
import asyncio
import logging

import execution
from app.node_replace_manager import NodeReplaceManager


class PromptServer:
    instance = None

    def __init__(self, loop: asyncio.AbstractEventLoop):
        PromptServer.instance = self

        self.loop = loop
        self.client_id = None
        self.last_prompt_id = None
        self.last_node_id = None
        self.sockets_metadata: dict = {}
        self.number = 0
        self.pending_sdapi: dict = {}

        self.node_replace_manager = NodeReplaceManager()
        self.prompt_queue = execution.PromptQueue(self)
        self.prompt_queue.add_task_done_callback(self._sdapi_task_done_callback)

    # ------------------------------------------------------------------ #
    # Interface expected by PromptExecutor / PromptQueue / custom nodes   #
    # ------------------------------------------------------------------ #

    def send_sync(self, event, data, sid=None):
        pass  # no WebSocket clients

    def queue_updated(self):
        pass

    def send_progress_text(self, text, node_id, sid=None):
        pass

    # ------------------------------------------------------------------ #
    # SDAPI completion signalling                                          #
    # ------------------------------------------------------------------ #

    def _sdapi_task_done_callback(self, prompt_id: str):
        event = self.pending_sdapi.get(prompt_id)
        if event is not None:
            self.loop.call_soon_threadsafe(event.set)
