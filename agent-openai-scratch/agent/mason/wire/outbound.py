"""Outbound stream translation: OpenAI Agents SDK stream events -> Responses stream events (dicts).

With the API set to ``responses``, the SDK emits ``raw_response_event``s whose ``.data`` is an
``openai.types.responses.ResponseStreamEvent`` — already the right shape, so they pass straight
through as dicts. The one thing the raw events do NOT carry is tool-call *outputs* (the return value
of a function tool) — those arrive as ``run_item_stream_event`` / ``tool_call_output_item`` — so we
surface them as an explicit ``response.output_item.done`` event.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any


async def process_agent_stream_events(
    async_stream: AsyncIterator[Any],
) -> AsyncGenerator[dict, None]:
    """Relay SDK raw response events as dicts; surface tool outputs the raw stream omits."""
    async for event in async_stream:
        if event.type == "raw_response_event":
            yield event.data.model_dump()
        elif event.type == "run_item_stream_event" and event.item.type == "tool_call_output_item":
            yield {"type": "response.output_item.done", "item": event.item.to_input_item()}
