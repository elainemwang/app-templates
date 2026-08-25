"""Agent server entry point. load_dotenv must run before agent imports (auth config)."""

# ruff: noqa: E402
import os
from pathlib import Path

from dotenv import load_dotenv

# Load env vars from .env before any other imports (agent needs auth config)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

from databricks_ai_bridge.long_running import LongRunningAgentServer

# Import the agent to register the @invoke/@stream functions, then configure global SDK state
# (agent client + tracing). configure() is a startup step, not an import side effect.
import agent.agent  # noqa: F401

agent.agent.configure()

# Pass the Lakebase autoscaling endpoint through from the env (set by the "postgres" app resource).
# When it's set, LongRunningAgentServer enables durable background mode + crash recovery; when unset
# (local dev, no Lakebase) it's None, so the server serves in-request over the local SQLite store.
agent_server = LongRunningAgentServer(
    "ResponsesAgent",
    db_autoscaling_endpoint=os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT"),
)

# Module-level app so uvicorn can import it by string (and to enable multiple workers).
app = agent_server.app


def main():
    agent_server.run(app_import_string="server.start_server:app")
