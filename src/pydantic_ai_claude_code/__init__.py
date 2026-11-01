"""Run Pydantic AI through a subscription-authenticated Claude Code CLI."""

from .diagnostics import doctor as doctor
from .exceptions import (
    ClaudeCodeCreditLimitError as ClaudeCodeCreditLimitError,
    ClaudeCodeQuotaError as ClaudeCodeQuotaError,
    ClaudeCodeSubscriptionLimitError as ClaudeCodeSubscriptionLimitError,
    ClaudeCodeUsageCreditsError as ClaudeCodeUsageCreditsError,
)
from .model import ClaudeCodeModel as ClaudeCodeModel, resolve_model as resolve_model
