"""FastAPI serving of the Responses API — the plumbing behind the agent's HTTP surface.

Hand-builds the Responses contract so the template shows exactly how a Databricks agent is served.
You edit the agent in ``agent/agent.py``; you rarely touch this file. It provides:

- ``POST /invocations`` and ``POST /responses`` — the OpenAI Responses API. ``stream: true`` returns
  an SSE stream (``data: {...}`` frames ending with ``data: [DONE]``); ``background: true`` returns a
  ``resp_...`` id immediately and runs the turn in the background.
- ``GET /responses/{id}`` — poll a background run's status/result.
- ``GET /health`` — liveness.
- An MLflow span around every request (inputs/outputs set), so tracing works when configured.

**Background mode here is in-memory and single-process** — a teaching stand-in, not durable. Runs
live in a dict in this process: they do NOT survive a restart and are NOT shared across replicas.
Production durability (crash recovery, cross-pod resume, surviving the ~120s Apps proxy timeout)
would need a shared durable store behind ``_BackgroundRuns``.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import mlflow
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

# Request keys that control transport; stripped before the request reaches the handler.
_STREAM_KEY = "stream"
_BACKGROUND_KEY = "background"
_MESSAGE_FORMAT_ATTR = "mlflow.message.format"

InvokeHandler = Callable[[ResponsesAgentRequest], Awaitable[ResponsesAgentResponse]]
StreamHandler = Callable[[ResponsesAgentRequest], AsyncGenerator[Any, None]]


def _sse(data: dict | str) -> str:
    return f"data: {json.dumps(data) if isinstance(data, dict) else data}\n\n"


def _as_dict(result: Any) -> dict:
    return result.model_dump() if hasattr(result, "model_dump") else dict(result)


class _BackgroundRuns:
    """In-memory store of background runs, keyed by response id. Single-process, non-durable."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}

    def create(self) -> str:
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self._runs[response_id] = {"status": "in_progress", "output": None, "error": None}
        return response_id

    def complete(self, response_id: str, output: dict) -> None:
        self._runs[response_id] = {"status": "completed", "output": output, "error": None}

    def fail(self, response_id: str, error: str) -> None:
        self._runs[response_id] = {"status": "failed", "output": None, "error": error}

    def get(self, response_id: str) -> dict | None:
        return self._runs.get(response_id)


def build_app(invoke_handler: InvokeHandler, stream_handler: StreamHandler) -> FastAPI:
    """Build the FastAPI app wiring the Responses API to the agent's invoke/stream handlers."""
    app = FastAPI(title="Agent Server")
    runs = _BackgroundRuns()

    async def _invoke(request: ResponsesAgentRequest) -> dict:
        with mlflow.start_span(name="invoke_handler") as span:
            span.set_inputs(request.model_dump())
            result = _as_dict(await invoke_handler(request))
            span.set_attribute(_MESSAGE_FORMAT_ATTR, "openai")
            span.set_outputs(result)
            return result

    async def _stream(request: ResponsesAgentRequest) -> AsyncGenerator[str, None]:
        with mlflow.start_span(name="stream_handler") as span:
            span.set_inputs(request.model_dump())
            chunks: list[dict] = []
            try:
                async for event in stream_handler(request):
                    chunk = _as_dict(event)
                    chunks.append(chunk)
                    yield _sse(chunk)
                span.set_attribute(_MESSAGE_FORMAT_ATTR, "openai")
                span.set_outputs(chunks)
            except Exception as e:  # surface the error in-band, then close the stream
                yield _sse({"error": str(e)})
            yield _sse("[DONE]")

    async def _run_background(response_id: str, request: ResponsesAgentRequest) -> None:
        try:
            runs.complete(response_id, await _invoke(request))
        except Exception as e:
            runs.fail(response_id, str(e))

    async def _handle(request: Request):
        data = await request.json()
        is_stream = bool(data.pop(_STREAM_KEY, False))
        is_background = bool(data.pop(_BACKGROUND_KEY, False))
        req = ResponsesAgentRequest(**data)

        if is_background:
            response_id = runs.create()
            # Fire-and-forget; the task updates `runs` when it finishes. Non-durable (in-memory).
            asyncio.create_task(_run_background(response_id, req))
            return JSONResponse({"id": response_id, "status": "in_progress", "object": "response"})
        if is_stream:
            return StreamingResponse(_stream(req), media_type="text/event-stream")
        return JSONResponse(await _invoke(req))

    app.add_api_route("/invocations", _handle, methods=["POST"])
    app.add_api_route("/responses", _handle, methods=["POST"])

    @app.get("/responses/{response_id}")
    async def retrieve(response_id: str):
        run = runs.get(response_id)
        if run is None:
            return JSONResponse({"error": "unknown response id"}, status_code=404)
        if run["status"] == "completed":
            # Spread the output first so our id/status win over the response's own (unset) fields.
            return JSONResponse({**run["output"], "id": response_id, "status": "completed"})
        return JSONResponse({"id": response_id, "status": run["status"], "error": run["error"]})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
