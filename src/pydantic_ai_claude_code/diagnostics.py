"""Quota-free diagnostics through the public Claude Agent SDK."""

from __future__ import annotations

import asyncio

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKError, ProcessError, ResultError
from pydantic import JsonValue, TypeAdapter
from pydantic_ai.exceptions import ModelAPIError, UserError

from ._auth import subscription_account
from ._transport import inspect_cli

_OBJECT = TypeAdapter(dict[str, JsonValue])


async def doctor(*, sdk_options: ClaudeAgentOptions | None = None) -> dict[str, JsonValue]:
    """Check SDK initialization and account classification without inference.

    Successful reports omit account identifiers. Failed reports preserve SDK
    exceptions, terminal payloads, stderr, and causes instead of redacting them.
    """
    try:
        async with asyncio.timeout(15):
            info = await inspect_cli(sdk_options)
            account = subscription_account(info)
    except (ClaudeSDKError, ModelAPIError, UserError, TimeoutError) as error:
        report: dict[str, JsonValue] = {
            'ok': False,
            'error': str(error),
            'error_type': type(error).__name__,
            'inference_performed': False,
        }
        if isinstance(error, ProcessError):
            report.update(exit_code=error.exit_code, stderr=error.stderr)
        if isinstance(error, ResultError):
            report['data'] = _OBJECT.validate_python(error.data, strict=True)
        if error.__cause__ is not None:
            report['cause'] = str(error.__cause__)
        try:
            report['notes'] = [note for note in error.__notes__]
        except AttributeError:
            pass
        return report
    return {
        'ok': True,
        'claude_version': info.get('claude_code_version'),
        'auth': {key: account.get(key) for key in ('apiProvider', 'subscriptionType')},
        'sdk_ready': True,
        'inference_performed': False,
    }
