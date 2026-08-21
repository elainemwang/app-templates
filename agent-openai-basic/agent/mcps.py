"""MCP servers for the agent.

Empty by default — the agent runs with no MCP servers. To add one, append an ``MCPServer`` (from the
OpenAI Agents SDK — e.g. ``MCPServerStreamableHttp``/``MCPServerSse``) to the list in
``build_mcp_servers``. ``connect`` opens them for the duration of a request via the SDK's
``MCPServerManager`` (connects on enter, cleans up on exit, drops any that fail), and the agent
calls their tools over those live connections until the run finishes.
"""

from contextlib import AsyncExitStack

from agents.mcp import MCPServer, MCPServerManager


def build_mcp_servers() -> list[MCPServer]:
    """Return the MCP servers to offer the agent. Empty by default — add your own.

    Example (Databricks-managed MCP over the workspace host, authed as the app SP):

        from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams
        from databricks.sdk import WorkspaceClient

        host = WorkspaceClient().config.host
        return [
            MCPServerStreamableHttp(
                params=MCPServerStreamableHttpParams(url=f"{host}/api/2.0/mcp/functions/system/ai"),
                name="system_ai",
            ),
        ]
    """
    return []


async def connect(stack: AsyncExitStack) -> list[MCPServer]:
    """Open the configured MCP servers for this request; returns the connected ones.

    No servers configured -> empty list. The ``MCPServerManager`` connects them on enter and cleans
    them up when ``stack`` exits at the end of the request.
    """
    servers = build_mcp_servers()
    if not servers:
        return []
    manager = await stack.enter_async_context(MCPServerManager(servers))
    return manager.active_servers
