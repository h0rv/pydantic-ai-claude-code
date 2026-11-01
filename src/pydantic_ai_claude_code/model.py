"""Claude Code model backed by the public Python Agent SDK.

The public SDK owns the CLI process. This backend does not intercept HTTP,
read credentials, or execute application tools inside Claude Code.
"""

from __future__ import annotations

import math
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import cached_property
from typing import ClassVar

import httpx2
from anthropic import APIConnectionError, AsyncAnthropic
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKError
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelConnectionError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import (
    ModelRequestParameters,
    ModelResolutionContext,
    StreamedResponse,
    check_allow_model_requests,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.profiles import ModelProfileSpec, ToolAdditionMode, ToolDeferralMode
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.settings import ModelSettingName, ModelSettings
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.usage import RequestUsage

from ._request import application_headers
from ._transport import ClaudeCodeTransport


class ClaudeCodeModel(AnthropicModel):
    """Run installed or SDK-bundled Claude Code through the public Agent SDK.

    Pydantic AI owns history, tools, structured output, and stream mapping.
    Claude Code owns authentication and model access. SDK errors are preserved;
    upstream HTTP metadata is unavailable unless the SDK exposes it.

    `sdk_options` is the SDK's own configuration, not an adapter-specific copy.
    `allow_usage_credits` accepts reported paid usage; it cannot prevent the
    first charge or change account-level billing settings.
    """

    unsupported_settings: ClassVar[frozenset[ModelSettingName]] = AnthropicModel.unsupported_settings | frozenset[
        ModelSettingName
    ]({'max_retries'})

    def __init__(
        self,
        model_name: str | None = None,
        *,
        sdk_options: ClaudeAgentOptions | None = None,
        idle_timeout: float = 300,
        total_timeout: float = 1800,
        allow_usage_credits: bool = False,
        settings: ModelSettings | None = None,
        profile: ModelProfileSpec | None = None,
    ) -> None:
        for timeout in (idle_timeout, total_timeout):
            if not math.isfinite(timeout) or timeout <= 0:
                raise UserError('Claude SDK timeouts must be finite positive seconds.')
        if (
            model_name is not None
            and sdk_options is not None
            and sdk_options.model is not None
            and model_name != sdk_options.model
        ):
            raise UserError('model_name and sdk_options.model must agree when both are provided.')
        if model_name is None:
            model_name = sdk_options.model if sdk_options is not None and sdk_options.model is not None else 'haiku'
        transport = ClaudeCodeTransport(sdk_options, idle_timeout, total_timeout, allow_usage_credits)
        client = AsyncAnthropic(
            api_key='local-sdk-placeholder',
            base_url='https://claude-agent-sdk.invalid',
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=transport, trust_env=False),
        )
        super().__init__(
            model_name, provider=AnthropicProvider(anthropic_client=client), settings=settings, profile=profile
        )

    supported_tool_addition_modes: ClassVar[frozenset[ToolAdditionMode]] = frozenset()
    supported_tool_deferral_modes: ClassVar[frozenset[ToolDeferralMode]] = frozenset()

    @cached_property
    def profile(self) -> AnthropicModelProfile:
        # Use core's instruction/tool-availability fallback instead of claiming
        # API roles that Claude Code's streaming input cannot represent.
        return {
            **super().profile,
            'supports_inline_system_prompts': False,
            'supports_request_scoped_system_prompts': False,
            'supported_native_tools': frozenset(),
        }

    @property
    def system(self) -> str:
        return 'claude-code'

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return frozenset()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        async with self.request_stream(messages, model_settings, model_request_parameters) as response:
            async for _ in response:
                pass
            result = response.get()
        return result

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[object] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        check_allow_model_requests()
        for layer in (self.settings, model_settings):
            if headers := (layer or {}).get('extra_headers'):
                application_headers(headers.items())
        if (model_settings or {}).get('max_retries', (self.settings or {}).get('max_retries', 0)):
            raise UserError('Automatic transport retries are disabled for subscription-backed requests.')
        if model_request_parameters.native_tools:
            raise UserError('Only Pydantic AI function and output tools may execute.')
        try:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as response:
                response.provider_details = {
                    **(response.provider_details or {}),
                    'transport': 'claude-agent-sdk',
                    'upstream_http_metadata_available': False,
                }
                yield response
        except ModelConnectionError as exc:
            cause = exc.__cause__
            if isinstance(cause, APIConnectionError) and isinstance(
                cause.__cause__, (ClaudeSDKError, UserError, UnexpectedModelBehavior)
            ):
                raise cause.__cause__
            raise

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        raise UserError('The Agent SDK does not expose a quota-free token counting endpoint.')


def resolve_model(ctx: ModelResolutionContext[AgentDepsT], model_id: str) -> ClaudeCodeModel | None:
    """Resolve claude-code:NAME through Pydantic AI's standard capability."""
    if not model_id.startswith('claude-code:'):
        return None
    return ClaudeCodeModel(model_id.removeprefix('claude-code:'))
