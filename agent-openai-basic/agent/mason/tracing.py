"""MLflow tracing setup — opt-in, gated on both ``MLFLOW_EXPERIMENT_ID`` and ``MLFLOW_TRACKING_URI``.

Requiring both avoids the half-configured case where traces silently export to a local file store
instead of the workspace. When unset, tracing is disabled outright so the agent-server framework's
mandatory per-request span has nothing to export to. No user decision lives here — it's all driven
by env — so this whole module is a candidate to move behind an SDK helper.
"""

import os

import mlflow
from agents.tracing import set_trace_processors

ENABLED = bool(os.getenv("MLFLOW_EXPERIMENT_ID") and os.getenv("MLFLOW_TRACKING_URI"))


def configure() -> None:
    """Wire up tracing. Call once at startup."""
    # Clear the Agents SDK's default trace processors so it doesn't export to OpenAI's backend;
    # MLflow's autolog (below) is the only trace sink we want.
    set_trace_processors([])
    if ENABLED:
        mlflow.openai.autolog()
    else:
        # The agent-server framework wraps every request in a span regardless; without an
        # experiment it would try to export to a missing one (INVALID_PARAMETER_VALUE), so disable.
        mlflow.tracing.disable()


def tag_session(session_id: str) -> None:
    """Tag the active MLflow trace with the session id, when tracing is enabled."""
    if ENABLED and session_id:
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
