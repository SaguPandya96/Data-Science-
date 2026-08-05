"""Model providers.

The mock provider is mandatory and offline; external providers are optional and
credential-gated. See ADR-003 for why that asymmetry is deliberate.
"""

from __future__ import annotations

from evalforge.providers.base import (
    Message,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from evalforge.providers.mock import MockModelProvider, TurnState
from evalforge.providers.registry import build_provider, provider_is_offline

__all__ = [
    "Message",
    "MockModelProvider",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "ToolCallRequest",
    "TurnState",
    "build_provider",
    "provider_is_offline",
]
