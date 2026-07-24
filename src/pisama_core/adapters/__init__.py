"""Platform adapters for Pisama.

Adapters provide the interface between pisama-core and specific
agent platforms (Claude Code, LangGraph, etc.).
"""

from pisama_core.adapters.base import (
    InjectionMethod,
    InjectionResult,
    PlatformAdapter,
)
from pisama_core.adapters.bedrock import parse_invoke_agent as parse_bedrock_invoke_agent
from pisama_core.adapters.deep_agents import DeepAgentsAdapter, parse_deep_agents_trace
from pisama_core.adapters.gemini import parse_interactions_response as parse_gemini_interactions
from pisama_core.adapters.google_adk import GoogleAdkAdapter, parse_adk_trace
from pisama_core.adapters.openai import (
    parse_assistants_run as parse_openai_assistants_run,
)
from pisama_core.adapters.openai import (
    parse_response as parse_openai_response,
)

__all__ = [
    "PlatformAdapter",
    "InjectionResult",
    "InjectionMethod",
    "parse_openai_assistants_run",
    "parse_openai_response",
    "parse_bedrock_invoke_agent",
    "DeepAgentsAdapter",
    "parse_deep_agents_trace",
    "parse_gemini_interactions",
    "GoogleAdkAdapter",
    "parse_adk_trace",
]
