"""Catchable Claude subscription and usage-credit outcomes."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import ClassVar

from claude_agent_sdk import RateLimitEvent, RateLimitInfo
from pydantic_ai.exceptions import ModelQuotaExceededError


class ClaudeCodeQuotaError(ModelQuotaExceededError):
    """A quota outcome identified by SDK billing metadata, not a guessed HTTP status.

    `event` retains the complete SDK rate-limit record, including unknown raw
    fields. `body` retains a terminal SDK error payload when one exists, and
    the original SDK exception remains the cause.
    """

    description: ClassVar[str] = 'Claude Code reported a quota limit.'

    def __init__(self, model_name: str, event: RateLimitEvent, *, body: object | None = None) -> None:
        self.event = event
        data = asdict(event)
        super().__init__(model_name, self.description + ' ' + json.dumps(data), body=data if body is None else body)

    @property
    def rate_limit_info(self) -> RateLimitInfo:
        """SDK limit type, reset timestamps, credit status, and original raw data."""
        return self.event.rate_limit_info


class ClaudeCodeSubscriptionLimitError(ClaudeCodeQuotaError):
    """The SDK identifies an exhausted subscription window."""

    description = 'Claude Code subscription limit reached.'


class ClaudeCodeCreditLimitError(ClaudeCodeQuotaError):
    """The SDK identifies a rejected usage-credit limit, even with local opt-in."""

    description = 'Claude Code usage-credit limit reached.'


class ClaudeCodeUsageCreditsError(ClaudeCodeQuotaError):
    """The SDK reports paid credit use, but the application has not opted in."""

    description = (
        'Claude SDK reported usage-credit spending. The first charge may already have occurred. '
        'Disable usage credits in your account to prevent charges, or explicitly set allow_usage_credits=True.'
    )
