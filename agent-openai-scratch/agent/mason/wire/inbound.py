"""Inbound request handling: pull the session id and prepare the OpenAI Agents SDK run input.

The request body is a plain dict shaped like the OpenAI Responses API — ``input`` is a list of
``openai.types.responses`` input items (what ``Runner.run`` accepts), plus an optional top-level
``session_id`` for multi-turn. No wrapper types: the SDK validates the input itself.
"""

from agents.memory.session import SessionABC
from uuid_utils import uuid7


def get_session_id(request: dict) -> str:
    """Return the request's ``session_id`` (for multi-turn), or a fresh UUID for a new conversation."""
    return str(request.get("session_id") or uuid7())


async def deduplicate_input(request: dict, session: SessionABC) -> list[dict]:
    """Return the input messages to pass to the Runner, avoiding duplication with session history.

    When a client sends the full conversation history AND the session already has that history
    persisted, passing everything through would duplicate messages. If the session already covers
    the prior turns, only the latest message is needed — the session prepends the rest automatically.
    """
    messages = list(request.get("input") or [])
    # Normalize assistant message content from string to the structured list the SDK expects.
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("type") == "message"
            and msg.get("role") == "assistant"
            and isinstance(msg.get("content"), str)
        ):
            msg["content"] = [{"type": "output_text", "text": msg["content"], "annotations": []}]
    session_items = await session.get_items()
    if len(session_items) >= len(messages) - 1:
        return [messages[-1]]
    return messages
