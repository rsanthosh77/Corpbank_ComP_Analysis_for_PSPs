"""Credential-safe progress: stages and timings, never prompts or tool payloads."""

import asyncio
from contextlib import asynccontextmanager, suppress
from time import monotonic

from langchain_core.callbacks import AsyncCallbackHandler


class ResearchProgress(AsyncCallbackHandler):
    def __init__(self, emit=None, usage_reader=None):
        super().__init__()
        self.emit = emit or (lambda text: print(text, flush=True))
        self.active = {}
        self.usage_reader = usage_reader

    def label(self, metadata):
        metadata = metadata or {}
        stage = metadata.get("research_stage", "compatibility_check")
        worker = metadata.get("research_worker")
        return f"{stage} / {worker}" if worker else stage

    async def on_custom_event(self, name, data, **kwargs):
        if name == "research_stage_started":
            self.emit(f"Started: {data['stage']}")
        elif name == "research_worker_validated":
            self.emit(f"Validated: {data['stage']} / {data['worker']}: "
                      f"{data['assessments']} assessments, {data['capabilities']} capabilities")

    async def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        label = self.label(metadata)
        self.active[run_id] = (f"{label}: waiting for model response", monotonic())
        self.emit(f"{label}: model request started")

    async def on_llm_end(self, response, *, run_id, **kwargs):
        entry = self.active.pop(run_id, None)
        if entry:
            label = entry[0].split(": waiting for", 1)[0]
            self.emit(f"{label}: model response received ({monotonic() - entry[1]:.0f}s)")
        if self.usage_reader:
            model = self.usage_reader()["model"]
            self.emit(f"Usage so far: {model['attempts']} model attempts; "
                      f"{model['input_tokens']} input / {model['output_tokens']} output tokens; "
                      f"{model['cache_hits']} cached responses")

    async def on_llm_error(self, error, *, run_id, **kwargs):
        self.active.pop(run_id, None)

    async def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kwargs):
        label = self.label(metadata)
        tool = serialized.get("name", "research tool")
        self.active[run_id] = (f"{label}: waiting for {tool}", monotonic())
        self.emit(f"{label}: {tool} started")

    async def on_tool_end(self, output, *, run_id, **kwargs):
        entry = self.active.pop(run_id, None)
        if entry:
            self.emit(f"{entry[0].replace('waiting for', 'received')} ({monotonic() - entry[1]:.0f}s)")

    async def on_tool_error(self, error, *, run_id, **kwargs):
        self.active.pop(run_id, None)

    @asynccontextmanager
    async def heartbeat(self, interval=30):
        async def pulse():
            while True:
                await asyncio.sleep(interval)
                for label, start in list(self.active.values())[:4]:
                    self.emit(f"Still working: {label} ({monotonic() - start:.0f}s)")

        task = asyncio.create_task(pulse())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
