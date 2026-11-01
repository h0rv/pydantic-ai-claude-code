"""SDK-only account diagnostics never submit generation frames."""

from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ProcessError, ResultError
from pydantic import JsonValue
from test_sdk import Harness, sdk as sdk

from pydantic_ai_claude_code import diagnostics, doctor
from pydantic_ai_claude_code._auth import subscription_account

pytestmark = pytest.mark.anyio


async def test_doctor__should_check_bundled_sdk_and_account_without_inference(sdk: Harness, tmp_path: Path) -> None:
    sdk.server_info['account']['email'] = 'identity-not-in-success-report@example.test'
    report = await doctor(sdk_options=ClaudeAgentOptions(cwd=tmp_path, env={'APP_CONTEXT': 'doctor'}))
    assert report == {
        'ok': True,
        'claude_version': '2.1.312',
        'auth': {'apiProvider': 'firstParty', 'subscriptionType': 'Claude Team'},
        'sdk_ready': True,
        'inference_performed': False,
    }
    assert sdk.transports == [] and len(sdk.preflights) == 1
    assert sdk.preflights[0].options.cli_path is None
    assert sdk.preflights[0].options.cwd == tmp_path
    assert sdk.preflights[0].options.env['APP_CONTEXT'] == 'doctor'
    assert sdk.preflights[0].closed.is_set() and sdk.preflights[0].reader_done.is_set()
    assert not any(frame['type'] in ('user', 'assistant') for frame in sdk.preflights[0].writes)


async def test_doctor__should_preserve_raw_sdk_error_details(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ProcessError('SDK process failure', exit_code=17, stderr='original stderr')
    error = ResultError('original SDK error', {'unknown': {'preserved': True}, 'api_error_status': 429}, exit_code=17)
    error.__cause__ = original
    error.add_note('original SDK note')

    async def failed(sdk_options: ClaudeAgentOptions | None = None) -> dict[str, str]:
        raise error

    monkeypatch.setattr(diagnostics, 'inspect_cli', failed)
    report = await doctor()
    assert report['ok'] is False and report['inference_performed'] is False
    assert report['data'] == error.data
    assert report['exit_code'] == 17
    assert report['cause'] == str(original)
    assert report['notes'] == ['original SDK note']


@pytest.mark.parametrize('plan', ['Claude Pro', 'Claude Max', 'Claude Ultra', 'Claude Team', 'Claude Enterprise'])
def test_subscription_account__should_accept_sdk_plan_names(plan: str) -> None:
    account: dict[str, JsonValue] = {'apiProvider': 'firstParty', 'subscriptionType': plan}
    assert subscription_account({'account': account}) == account


def test_source__should_not_contain_a_cli_supervisor_or_http_relay() -> None:
    root = Path(diagnostics.__file__).parent
    source = '\n'.join(path.read_text(encoding='utf-8') for path in root.glob('*.py'))
    for operation in (
        'create_subprocess_exec(',
        'create_subprocess_shell(',
        'subprocess.Popen(',
        'asyncio.start_server(',
    ):
        assert operation not in source
    assert not (root / '_relay.py').exists()
    assert not (root / 'sdk_model.py').exists()
