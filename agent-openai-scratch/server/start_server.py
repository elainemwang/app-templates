"""Agent server entry point — a raw FastAPI app, no serving framework.

The Responses API surface (routes, SSE framing, tracing spans, in-memory background mode) is built
by ``agent/mason/wire/serve.py``; this file just loads config, wires the agent handlers in, and runs
uvicorn.
"""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Importing the agent is side-effect-free (no env is read until configure()), so it sits up top.
import agent.agent
from agent.mason.wire.serve import build_app

# Load .env before the runtime steps below read env (agent client auth + tracing config).
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

agent.agent.configure()

# Module-level app so uvicorn can import it by string (and to enable multiple workers).
app = build_app(agent.agent.invoke_handler, agent.agent.stream_handler)


def main():
    uvicorn.run("server.start_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
