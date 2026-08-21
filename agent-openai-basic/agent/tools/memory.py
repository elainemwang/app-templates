"""Long-term memory tools — opt-in, gated on ``AGENT_MEMORY_STORE``.

Unlike the session store (short-term transcript for one conversation), long-term memory is exposed
to the model as two tools — ``remember`` and ``recall`` — over the Databricks managed memory store's
``agents/v1`` entries API. Facts persist across conversations. When ``AGENT_MEMORY_STORE`` is unset
the tools are not registered, so the agent runs without long-term memory.

Memory entries are per-actor. This uses the store name as the actor id, giving the agent one shared
long-term memory; change ``_ACTOR_ID`` to scope per user (e.g. from request context) if needed.
"""

import os

from agents import function_tool
from databricks.sdk import WorkspaceClient

_MEMORY_STORE = os.getenv("AGENT_MEMORY_STORE")
_AGENTS_V1 = "/api/agents/v1"
_ACTOR_ID = os.getenv("AGENT_MEMORY_ACTOR_ID", "agent")

if _MEMORY_STORE:
    _store_path = f"{_AGENTS_V1}/memory-stores/{_MEMORY_STORE}"

    def _api():
        # Build the client lazily (needs workspace auth) so importing this module stays cheap.
        return WorkspaceClient().api_client

    @function_tool
    def remember(fact: str, topic: str) -> str:
        """Persist a durable fact about the user in long-term memory."""
        _api().do(
            "POST",
            f"{_store_path}/entries",
            body={"actor_id": _ACTOR_ID, "path": f"/{topic}/{fact[:8]}.md", "content": fact},
        )
        return "stored"

    @function_tool
    def recall(query: str) -> str:
        """Search the user's long-term memory for facts relevant to ``query``."""
        data = _api().do(
            "POST",
            f"{_store_path}/entries:search",
            body={"actor_id": _ACTOR_ID, "query": query, "limit": 5},
        )
        entries = data.get("managed_memory_entries") or []
        return "\n".join(f"- {e.get('content')}" for e in entries) or "No relevant memories."
