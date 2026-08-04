"""Provider construction.

One function decides which provider a run uses. Keeping that decision in a single place
means the CLI, the orchestrator and the judge all resolve providers identically, and the
"no credentials" path produces the same helpful message everywhere.
"""

from __future__ import annotations

from evalforge.config import EvalForgeConfig
from evalforge.exceptions import ProviderUnavailableError
from evalforge.providers.base import ModelProvider
from evalforge.providers.mock import MockModelProvider

#: Providers that never require credentials.
OFFLINE_PROVIDERS = frozenset({"mock"})


def build_provider(
    config: EvalForgeConfig,
    provider_name: str | None = None,
    profile_name: str = "baseline",
) -> ModelProvider:
    """Construct the requested provider.

    Args:
        config: Effective configuration.
        provider_name: Override for ``config.provider.name``.
        profile_name: Behaviour profile, used only by the mock provider.

    Returns:
        A provider satisfying :class:`~evalforge.providers.base.ModelProvider`.

    Raises:
        ProviderUnavailableError: If the requested provider is unknown, or is an
            external provider whose credentials or SDK are absent.
    """
    name = (provider_name or config.provider.name).lower()

    if name == "mock":
        profile = config.failure_injection.profile(profile_name)
        return MockModelProvider(profile=profile, model_name=f"mock-{profile_name}-v1")

    if name == "anthropic":
        from evalforge.providers.external import AnthropicModelProvider

        model = config.provider.model
        if model.startswith("mock"):
            model = "claude-sonnet-5"
        return AnthropicModelProvider(model=model, max_retries=config.provider.max_retries)

    if name == "openai":
        from evalforge.providers.external import OpenAICompatibleProvider

        model = config.provider.model
        if model.startswith("mock"):
            model = "gpt-4o-mini"
        return OpenAICompatibleProvider(model=model, max_retries=config.provider.max_retries)

    raise ProviderUnavailableError(
        f"Unknown provider {name!r}. Supported: mock, anthropic, openai. "
        "The mock provider requires no credentials and powers the full offline demo."
    )


def provider_is_offline(provider_name: str) -> bool:
    """Whether a provider needs no network access or credentials."""
    return provider_name.lower() in OFFLINE_PROVIDERS
