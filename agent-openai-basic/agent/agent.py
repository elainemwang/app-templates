import os
from contextlib import AsyncExitStack
from typing import AsyncGenerator

import mlflow
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent import mcps
from agent.session_store import create_session
from agent.tools import all_tools  # importing the package auto-registers every tool module
from agent.wire.inbound import deduplicate_input, get_session_id
from agent.wire.outbound import process_agent_stream_events


# MLflow tracing is optional, enabled only when both a destination and an experiment are set (see
# configure()). Requiring both avoids the half-configured case where traces silently export to a
# local file store instead of the workspace.
_TRACING_ENABLED = bool(os.getenv("MLFLOW_EXPERIMENT_ID") and os.getenv("MLFLOW_TRACKING_URI"))


def configure() -> None:
    """Wire up global agent-SDK state; call once at server startup."""
    set_default_openai_client(AsyncDatabricksOpenAI())
    set_default_openai_api("responses")
    set_trace_processors([])
    # The agent-server framework wraps every request in a span regardless. Without an experiment it
    # would try to export to a missing one (INVALID_PARAMETER_VALUE: experiment_id is missing), so
    # disable tracing outright when unconfigured to keep local runs quiet.
    if _TRACING_ENABLED:
        mlflow.openai.autolog()
    else:
        mlflow.tracing.disable()


def _tag_session(session_id: str) -> None:
    """Tag the active MLflow trace with the session id, when tracing is enabled."""
    if _TRACING_ENABLED and session_id:
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})


def create_agent(mcp_servers: list | None = None) -> Agent:
    return Agent(
        name="Agent",
        instructions="You are a helpful assistant.",
        model="databricks-gpt-5-2",
        tools=all_tools(),
        mcp_servers=mcp_servers or [],
    )


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    session_id = get_session_id(request)
    _tag_session(session_id)
    session = create_session(session_id)

    async with AsyncExitStack() as stack:
        agent = create_agent(mcp_servers=await mcps.connect(stack))
        messages = await deduplicate_input(request, session)
        result = await Runner.run(agent, messages, session=session)
    return ResponsesAgentResponse(
        output=[item.to_input_item() for item in result.new_items],
        custom_outputs={"session_id": session.session_id},
    )


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    session_id = get_session_id(request)
    _tag_session(session_id)
    session = create_session(session_id)

    async with AsyncExitStack() as stack:
        agent = create_agent(mcp_servers=await mcps.connect(stack))
        messages = await deduplicate_input(request, session)
        result = Runner.run_streamed(agent, input=messages, session=session)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
