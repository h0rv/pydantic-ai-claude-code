"""Explicitly gated subscription-consuming integration test."""

import os
from uuid import uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.usage import UsageLimits

from pydantic_ai_claude_code import ClaudeCodeModel

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get('PYDANTIC_AI_CLAUDE_CODE_LIVE') != 'I_ACCEPT_SUBSCRIPTION_USAGE',
        reason='Explicit opt-in required for two subscription-consuming model requests.',
    ),
]


class Answer(BaseModel):
    value: str


async def test_live__should_execute_python_tool_and_return_validated_output() -> None:
    value = uuid4().hex
    calls: list[str] = []
    async with ClaudeCodeModel('haiku') as model:
        agent = Agent(model, output_type=Answer, retries=0)

        @agent.tool_plain
        def read_local_value() -> str:
            """Read a value available only in this Python process."""
            calls.append('python')
            return value

        with override_allow_model_requests(True):
            result = await agent.run(
                'Call read_local_value, then return exactly its value in the typed output.',
                usage_limits=UsageLimits(request_limit=2),
            )
    assert calls == ['python']
    assert result.output == Answer(value=value)
    assert result.usage.requests == 2
