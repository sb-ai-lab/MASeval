"""Langfuse client initialization for judge traces."""

import os
from langfuse import Langfuse

# Initialize Langfuse client for uploading maseval judge traces
# This sets the global OpenTelemetry tracer provider for pydantic AI instrumentation
_langfuse_judge_client = None
_instrumentation_initialized = False


def get_langfuse_judge_client() -> Langfuse:
    """Get or create the Langfuse client for uploading maseval judge traces.

    This client uses LANGFUSE_PUBLIC_KEY_JUDGE and LANGFUSE_SECRET_KEY_JUDGE
    environment variables. It sets the global OpenTelemetry tracer provider,
    which enables pydantic AI agents to automatically upload traces.

    IMPORTANT: Call this AFTER you've finished downloading traces with the
    download client, to ensure the judge client's tracer provider is active
    when evaluations run.

    Returns:
        Langfuse: The initialized Langfuse client for judge traces
    """
    global _langfuse_judge_client, _instrumentation_initialized

    if _langfuse_judge_client is None:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY_JUDGE")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY_JUDGE")
        host = os.getenv("LANGFUSE_HOST_JUDGE")

        # Initialize with judge keys for uploading traces
        # This sets the global OpenTelemetry tracer provider
        _langfuse_judge_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

        # Enable instrumentation after judge client is initialized
        if not _instrumentation_initialized:
            from pydantic_ai import Agent

            Agent.instrument_all()
            _instrumentation_initialized = True

    return _langfuse_judge_client
