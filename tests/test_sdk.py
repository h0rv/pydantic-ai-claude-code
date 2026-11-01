"""Offline protocol tests using the real SDK client, parser, hooks, and MCP routing."""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
    Message,
    ProcessError,
    ResultError,
    Transport,
    query,
)
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, ToolOutput
from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.exceptions import (
    ContentFilterError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelTimeoutError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, ModelResolutionContext, override_allow_model_requests
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.tools import ToolDefinition

import pydantic_ai_claude_code._transport as sdk_transport
from pydantic_ai_claude_code import (
    ClaudeCodeCreditLimitError,
    ClaudeCodeModel,
    ClaudeCodeQuotaError,
    ClaudeCodeSubscriptionLimitError,
    ClaudeCodeUsageCreditsError,
    resolve_model,
)

pytestmark = pytest.mark.anyio
FIXTURES = Path(__file__).parent / 'fixtures'
PREFIX = 'mcp__pydantic_ai__'


def frames(name: str = 'text') -> list[dict[str, Any]]:
    return [json.loads(line) for line in (FIXTURES / f'{name}-stream.jsonl').read_text(encoding='utf-8').splitlines()]


def result_frame(**updates: Any) -> dict[str, Any]:
    return {**frames()[-1], **updates}


class ProtocolTransport(Transport):
    """Raw JSON I/O only. Nothing replaces ClaudeSDKClient or its control machinery."""

    def __init__(self, options: ClaudeAgentOptions, scenario: str, server_info: dict[str, Any]) -> None:
        self.options, self.scenario, self.server_info = options, scenario, server_info
        self.queue: asyncio.Queue[dict[str, Any] | Exception | None] = asyncio.Queue()
        self.writes: list[dict[str, Any]] = []
        self.catalog: list[dict[str, Any]] = []
        self.initialize: dict[str, Any] = {}
        self.closed, self.reader_done, self.started = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.close_started, self.release_close = asyncio.Event(), asyncio.Event()
        self.release_close.set()
        self.inventory_ready = asyncio.Event()
        self.deferred: dict[str, Any] | None = None
        self.terminal_result: dict[str, Any] | None = None
        self.process_error = ProcessError('synthetic SDK crash', exit_code=17, stderr='fixture stderr')

    def emit(self, frame: dict[str, Any] | Exception | None) -> None:
        self.queue.put_nowait(frame)

    def request(self, request_id: str, **request: Any) -> None:
        self.emit(dict(type='control_request', request_id=request_id, request=request))

    def mcp(self, request_id: str, **message: Any) -> None:
        self.request(
            request_id, subtype='mcp_message', server_name='pydantic_ai', message=dict(jsonrpc='2.0', **message)
        )

    async def connect(self) -> None:
        pass

    def is_ready(self) -> bool:
        return not self.closed.is_set()

    async def end_input(self) -> None:
        if self.scenario == 'hang':
            return
        self.emit(
            self.process_error
            if self.scenario in ('errors', 'status', 'subscription-status', 'credit-status')
            else None
        )

    async def close(self) -> None:
        self.close_started.set()
        await self.release_close.wait()
        self.closed.set()

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        try:
            while True:
                frame = await self.queue.get()
                if isinstance(frame, Exception):
                    raise frame
                if frame is None:
                    return
                yield frame
        finally:
            self.reader_done.set()

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            frame = json.loads(line)
            self.writes.append(frame)
            if frame['type'] == 'control_request':
                assert frame['request']['subtype'] == 'initialize'
                self.initialize = frame['request']
                response = dict(subtype='success', request_id=frame['request_id'], response=self.server_info)
                self.emit(dict(type='control_response', response=response))
                if isinstance(self.options.mcp_servers, dict) and 'pydantic_ai' in self.options.mcp_servers:
                    params = dict(
                        protocolVersion='2024-11-05', capabilities={}, clientInfo=dict(name='offline-test', version='1')
                    )
                    self.mcp('mcp-init', id=0, method='initialize', params=params)
                else:
                    self.inventory_ready.set()
            elif frame['type'] == 'control_response':
                response = frame['response']
                assert response['subtype'] == 'success', response
                match response['request_id']:
                    case 'mcp-init':
                        assert 'result' in response['response']['mcp_response']
                        self.mcp('mcp-ready', method='notifications/initialized')
                    case 'mcp-ready':
                        self.mcp('catalog', id=1, method='tools/list', params={})
                    case 'catalog':
                        self.catalog = response['response']['mcp_response']['result']['tools']
                        self.inventory_ready.set()
                    case 'proposal':
                        assert response['response']['hookSpecificOutput']['permissionDecision'] == 'defer'
                        self.emit(result_frame(result='', deferred_tool_use=self.deferred))
                    case _:
                        pytest.fail(f'Unexpected control response: {response}')
            elif frame['type'] in ('user', 'assistant'):
                await self.inventory_ready.wait()
                if frame['type'] == 'assistant':
                    continue  # Assistant history never triggers inference or needs a result ack.
                if frame.get('shouldQuery') is False:
                    self.emit(result_frame(num_turns=0, result='', usage={}))
                    continue
                self.started.set()
                init = frames()[0]
                assert isinstance(self.options.mcp_servers, dict)
                servers = [dict(name=name, status='connected') for name in self.options.mcp_servers]
                init.update(
                    tools=[PREFIX + tool['name'] for tool in self.catalog],
                    model=self.options.model,
                    mcp_servers=servers,
                )
                self.emit(init)
                self.generate()
            else:
                pytest.fail(f'Unexpected SDK stdin frame: {frame}')

    def generate(self) -> None:  # noqa: C901 - independent adversarial protocol scenarios.
        if self.scenario == 'hang':
            return
        if self.scenario == 'process':
            self.emit(self.process_error)
            return
        if self.scenario in (
            'rate-limit',
            'quota',
            'quota-rejected',
            'credit-limit',
            'subscription-status',
            'credit-status',
            'unknown-rate-limit',
        ):
            rate = next(frame for frame in frames() if frame['type'] == 'rate_limit_event')
            rate['rate_limit_info'].update(
                status='allowed' if self.scenario == 'quota' else 'rejected',
                rateLimitType='overage' if self.scenario in ('credit-limit', 'credit-status') else 'seven_day',
                isUsingOverage=self.scenario in ('quota', 'quota-rejected'),
                overageInUse=self.scenario in ('quota', 'quota-rejected'),
                future_limit_field={'diagnostic': 'synthetic-sensitive-limit-diagnostic'},
            )
            if self.scenario == 'unknown-rate-limit':
                rate['rate_limit_info']['rateLimitType'] = None
            self.emit(rate)
        if self.scenario in ('errors', 'status', 'subscription-status', 'credit-status'):
            error = result_frame(is_error=True, result='', errors=['fixture terminal failure'])
            error['usage']['future_usage_field'] = {'diagnostic': 'synthetic-sensitive-diagnostic'}
            error['permission_denials'] = [{'tool_name': 'fixture', 'tool_input': {'secret': 'synthetic-secret'}}]
            if self.scenario != 'errors':
                error.update(errors=['synthetic-unredacted-api-error'], api_error_status=429)
            error['future_result_field'] = {'diagnostic': 'synthetic-unredacted-future-result'}
            self.terminal_result = error
            self.emit(error)
            return
        tool = self.scenario in ('tool', 'output-tool', 'unknown-tool')
        name = PREFIX + ('final_result' if self.scenario == 'output-tool' else 'return_value')
        if self.scenario == 'unknown-tool':
            name = PREFIX + 'undeclared_tool'
        for frame in frames('tool' if tool else 'text'):
            if frame['type'] != 'stream_event':
                continue
            frame = json.loads(json.dumps(frame).replace('mcp__probe__return_value', name))
            event = frame['event']
            event['future_native_field'] = {'preserved': True}  # SDK must not reject unknown native fields.
            if self.scenario == 'subagent':
                frame['parent_tool_use_id'] = 'toolu_forbidden_subagent'
            if self.scenario == 'missing-stop' and event['type'] == 'message_stop':
                continue
            if self.scenario == 'truncated' and event['type'] == 'content_block_stop':
                self.emit(None)
                return
            if self.scenario == 'refusal' and event['type'] == 'message_delta':
                event['delta']['stop_reason'] = 'refusal'
            if self.scenario == 'native' and event['type'] == 'content_block_delta':
                event['delta']['text'] = '{"value":"MCP_BRIDGE_OK"}' if event['delta']['text'] == 'PRINT_TRAN' else ''
            if self.scenario == 'thinking' and 'index' in event:
                event['index'] += 1
            self.emit(frame)
            if self.scenario == 'duplicate-response' and event['type'] == 'message_start':
                self.emit(frame)
            if self.scenario == 'restarted-stream' and event['type'] == 'message_stop':
                self.emit(next(item for item in frames() if item.get('event', {}).get('type') == 'message_start'))
            if self.scenario == 'future-fields' and event['type'] == 'message_start':
                self.emit({**frame, 'event': {'type': 'future_native_event', 'opaque': {'preserved': True}}})
            if self.scenario == 'thinking' and event['type'] == 'message_start':
                for thought in [
                    dict(type='content_block_start', index=0, content_block=dict(type='thinking', thinking='')),
                    dict(type='content_block_delta', index=0, delta=dict(type='thinking_delta', thinking='reasoning')),
                    dict(type='content_block_delta', index=0, delta=dict(type='signature_delta', signature='signed')),
                    dict(type='content_block_stop', index=0),
                ]:
                    self.emit({**frame, 'event': thought})
        if tool:
            self.deferred = dict(id='toolu_fake_001', name=name, input={'value': 'MCP_BRIDGE_OK'})
            callback = self.initialize['hooks']['PreToolUse'][0]['hookCallbackIds'][0]
            hook_input = dict(
                hook_event_name='PreToolUse',
                tool_name=name,
                tool_input=self.deferred['input'],
                tool_use_id=self.deferred['id'],
                session_id='fixture',
                transcript_path='',
                cwd='/offline',
            )
            self.request(
                'proposal',
                subtype='hook_callback',
                callback_id=callback,
                tool_use_id=self.deferred['id'],
                input=hook_input,
            )
        else:
            if self.scenario == 'missing-result':
                self.emit(None)
                return
            final = result_frame()
            if self.scenario == 'native':
                final['structured_output'] = {'value': 'MCP_BRIDGE_OK'}
            self.emit(final)


class Harness:
    def __init__(self) -> None:
        self.scenarios = ['text']
        self.server_info: dict[str, Any] = {
            'account': {'apiProvider': 'firstParty', 'subscriptionType': 'Claude Team'},
            'claude_code_version': '2.1.312',
        }
        self.block_close = False
        self.transports: list[ProtocolTransport] = []
        self.preflights: list[ProtocolTransport] = []

    def client(self, options: ClaudeAgentOptions) -> ClaudeSDKClient:
        transport = ProtocolTransport(options, 'preflight', self.server_info)
        self.preflights.append(transport)
        return ClaudeSDKClient(options=options, transport=transport)

    def query(self, prompt: AsyncIterable[dict[str, Any]], options: ClaudeAgentOptions) -> AsyncIterator[Message]:
        transport = ProtocolTransport(options, self.scenarios.pop(0), self.server_info)
        if self.block_close:
            transport.release_close.clear()
        self.transports.append(transport)
        return query(prompt=prompt, options=options, transport=transport)


@pytest.fixture
async def sdk(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Harness]:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('Offline SDK tests must not open sockets or launch processes')

    for target, names in [
        (socket.socket, ['connect', 'bind']),
        (subprocess, ['Popen']),
        (asyncio, ['create_subprocess_exec', 'create_subprocess_shell']),
    ]:
        for name in names:
            monkeypatch.setattr(target, name, forbidden)
    harness = Harness()
    monkeypatch.setattr(sdk_transport, '_client', harness.client)
    monkeypatch.setattr(sdk_transport, '_query', harness.query)
    with override_allow_model_requests(True):
        yield harness


@pytest.fixture
async def model(sdk: Harness) -> AsyncIterator[AnthropicModel]:
    instance = ClaudeCodeModel(
        'haiku',
        sdk_options=ClaudeAgentOptions(cli_path='/never-launch-claude', max_turns=1),
        idle_timeout=2,
        total_timeout=5,
        profile={'supports_json_schema_output': True, 'supports_inline_system_prompts': True},
        settings={'max_tokens': 64},
    )
    assert isinstance(instance, AnthropicModel)
    try:
        yield instance
    finally:
        await instance.client.close()


@pytest.mark.parametrize('stream', [False, True])
async def test_text__should_use_native_mappers_and_usage(model: AnthropicModel, sdk: Harness, stream: bool) -> None:
    agent = Agent(model)
    if stream:
        async with agent.run_stream('hello') as run:
            chunks = [chunk async for chunk in run.stream_text(delta=True)]
            assert ''.join(chunks) == 'PRINT_TRANSPORT_OK'
            usage = run.usage
    else:
        result = await agent.run('hello')
        assert result.output == 'PRINT_TRANSPORT_OK'
        usage = result.usage
    assert (usage.input_tokens, usage.output_tokens) == (15, 11)
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (7, 5)
    assert sdk.transports[0].closed.is_set() and sdk.transports[0].reader_done.is_set()
    options = sdk.transports[0].options
    assert options.cli_path == '/never-launch-claude' and options.tools == [] and options.max_turns == 1
    assert options.include_partial_messages
    assert all(
        frame.get('client_composed') is True
        for frame in sdk.transports[0].writes
        if frame['type'] in ('user', 'assistant')
    )


async def test_history__should_replay_native_frames_and_json_schema(model: AnthropicModel, sdk: Harness) -> None:
    schema = dict(
        type='object',
        properties={'value': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}},
        required=['value'],
        additionalProperties=False,
    )
    thinking = ThinkingPart('signed reasoning', signature='signed-fixture', provider_name=model.system)
    call = ToolCallPart('return_value', {'value': None}, tool_call_id='toolu_history')
    returned = ToolReturnPart('return_value', 'prior result', tool_call_id='toolu_history')
    history = [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[thinking, call]),
        ModelRequest(parts=[returned, UserPromptPart('continue')]),
    ]
    definition = ToolDefinition(name='return_value', parameters_json_schema=schema)
    await model.request(history, None, ModelRequestParameters(function_tools=[definition]))
    transport = sdk.transports[0]
    assert transport.catalog[0]['inputSchema'] == schema
    native = [frame for frame in transport.writes if frame['type'] in ('user', 'assistant')]
    assert [frame['message']['role'] for frame in native] == ['user', 'assistant', 'user']
    assert native[0]['shouldQuery'] is False and native[-1].get('shouldQuery') is not False
    blocks = native[1]['message']['content']
    assert blocks[0] == dict(type='thinking', thinking='signed reasoning', signature='signed-fixture')
    assert blocks[1]['name'] == PREFIX + 'return_value' and blocks[1]['input'] == {'value': None}
    assert native[-1]['message']['content'][0]['tool_use_id'] == 'toolu_history'


async def test_thinking__should_preserve_signed_blocks_and_honest_metadata(model: AnthropicModel, sdk: Harness) -> None:
    sdk.scenarios = ['thinking']
    response = await model.request([ModelRequest(parts=[UserPromptPart('think')])], None, ModelRequestParameters())
    thinking = response.parts[0]
    assert isinstance(thinking, ThinkingPart)
    assert (thinking.content, thinking.signature) == ('reasoning', 'signed')
    assert response.parts[1] == TextPart('PRINT_TRANSPORT_OK')
    assert response.provider_details is not None
    assert response.provider_details['upstream_http_metadata_available'] is False


class Answer(BaseModel):
    value: str


@pytest.mark.parametrize('mode', ['native', 'tool'])
async def test_structured_output__should_map_native_schema_and_output_tool(
    model: AnthropicModel,
    sdk: Harness,
    mode: str,
) -> None:
    sdk.scenarios = ['native' if mode == 'native' else 'output-tool']
    output = NativeOutput(Answer) if mode == 'native' else ToolOutput(Answer, name='final_result')
    result = await Agent(model, output_type=output).run('return a value')
    assert result.output == Answer(value='MCP_BRIDGE_OK')
    if mode == 'native':
        output_format = json.loads(sdk.transports[0].options.env['CLAUDE_CODE_EXTRA_BODY'])['output_config']['format']
        assert output_format['type'] == 'json_schema'
        assert output_format['schema']['properties']['value']['type'] == 'string'
        assert output_format['schema']['required'] == ['value']
    else:
        assert sdk.transports[0].deferred is not None


async def test_python_tool__should_execute_exactly_once_in_agent_loop(model: AnthropicModel, sdk: Harness) -> None:
    sdk.scenarios = ['tool', 'text']
    calls: list[str] = []
    agent = Agent(model)

    @agent.tool_plain
    def return_value(value: str) -> str:
        assert sdk.transports[0].closed.is_set(), 'SDK must finish before Python tools execute'
        calls.append(value)
        return value

    result = await agent.run('use the tool')
    assert calls == ['MCP_BRIDGE_OK'] and result.output == 'PRINT_TRANSPORT_OK'
    assert result.usage.requests == 2 and len(sdk.transports) == 2
    assert len(sdk.preflights) == 1, 'Account preflight must be cached on the model transport'
    assert sdk.preflights[0].closed.is_set() and sdk.preflights[0].reader_done.is_set()
    replay = [frame for frame in sdk.transports[1].writes if frame['type'] == 'user']
    assert any(
        block.get('type') == 'tool_result' and block['tool_use_id'] == 'toolu_fake_001'
        for frame in replay
        for block in frame['message']['content']
    )


@pytest.mark.parametrize('scenario', ['errors', 'status', 'process'])
async def test_sdk_failure__should_preserve_sdk_errors_and_stderr(
    model: AnthropicModel, sdk: Harness, scenario: str
) -> None:
    sdk.scenarios = [scenario]
    with pytest.raises(ProcessError) as caught:
        await Agent(model).run('fail')
    error = caught.value
    if scenario == 'process':
        assert error is sdk.transports[0].process_error
        assert error.exit_code == 17 and error.stderr == 'fixture stderr'
    else:
        assert isinstance(error, ResultError)
        assert error.api_error_status == (429 if scenario == 'status' else None)
        assert error.errors == (
            ['synthetic-unredacted-api-error'] if scenario == 'status' else ['fixture terminal failure']
        )
        assert error.data['usage']['future_usage_field'] == {'diagnostic': 'synthetic-sensitive-diagnostic'}
        assert error.data['permission_denials'] == [
            {'tool_name': 'fixture', 'tool_input': {'secret': 'synthetic-secret'}}
        ]
        assert error.data['uuid'] == result_frame()['uuid']
        assert error.data['terminal_reason'] == 'completed'
        assert error.data['duration_api_ms'] == 1
        assert error.data == sdk.transports[0].terminal_result
        assert error.__cause__ is sdk.transports[0].process_error
        assert 'headers' not in error.data  # The SDK has no upstream HTTP headers.
    assert sdk.transports[0].closed.is_set() and sdk.transports[0].reader_done.is_set()


async def test_system_history__should_use_core_fallback_without_flattening_conversation(
    model: AnthropicModel, sdk: Harness
) -> None:
    history = [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('answer')]),
        ModelRequest(parts=[SystemPromptPart('changed instruction'), UserPromptPart('continue')]),
    ]
    parameters = ModelRequestParameters()
    response = await model.request(model.prepare_messages(history, parameters), None, parameters)
    assert response.parts == [TextPart('PRINT_TRANSPORT_OK')]
    sent = [frame for frame in sdk.transports[0].writes if frame['type'] in ('user', 'assistant')]
    assert [frame['message']['role'] for frame in sent] == ['user', 'assistant', 'user']
    assert 'changed instruction' in json.dumps(sent)
    assert model.profile.get('supports_inline_system_prompts') is False


async def test_tool_availability__should_use_current_tool_schemas_without_unsupported_inline_role(
    model: AnthropicModel, sdk: Harness
) -> None:
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='lookup', defer_loading=True), ToolDefinition(name='visible_helper')],
        revealed_tool_names={'lookup'},
    )
    history = [
        ModelRequest(parts=[UserPromptPart('First question.')]),
        ModelResponse(parts=[TextPart('A tool becomes available.')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['lookup']), UserPromptPart('Continue.')]),
    ]
    await model.request(model.prepare_messages(history, parameters), None, parameters)
    transport = sdk.transports[0]
    tools = transport.catalog
    lookup = next(tool for tool in tools if tool['name'] == 'lookup')
    assert lookup.get('defer_loading') is not True
    assert model.tool_addition_mode is None
    sent = [frame for frame in transport.writes if frame['type'] in ('user', 'assistant')]
    assert [frame['message']['role'] for frame in sent] == ['user', 'assistant', 'user']


async def test_cancellation__should_join_sdk_reader_and_close(model: AnthropicModel, sdk: Harness) -> None:
    sdk.scenarios = ['hang']
    with anyio.fail_after(8):
        async with anyio.create_task_group() as group:
            group.start_soon(Agent(model).run, 'wait forever')
            while not sdk.transports:
                await anyio.lowlevel.checkpoint()
            transport = sdk.transports[0]
            await transport.started.wait()
            transport.release_close.clear()
            group.cancel_scope.cancel()
            with anyio.CancelScope(shield=True):
                await transport.close_started.wait()
                assert not transport.closed.is_set()
                transport.release_close.set()
    assert sdk.transports[0].closed.is_set() and sdk.transports[0].reader_done.is_set()


@pytest.mark.parametrize(
    'scenario',
    ['subagent', 'duplicate-response', 'restarted-stream', 'truncated', 'missing-stop', 'missing-result'],
)
async def test_invalid_stream__should_fail_and_close_without_retry(
    model: AnthropicModel, sdk: Harness, scenario: str
) -> None:
    sdk.scenarios = [scenario]
    with anyio.fail_after(5), pytest.raises(UnexpectedModelBehavior):
        await Agent(model).run('reject this invalid stream')
    assert len(sdk.transports) == 1
    assert sdk.transports[0].closed.is_set() and sdk.transports[0].reader_done.is_set()


async def test_unknown_tool__should_fail_before_python_execution(model: AnthropicModel, sdk: Harness) -> None:
    sdk.scenarios = ['unknown-tool']
    calls: list[str] = []
    agent = Agent(model)

    @agent.tool_plain
    def return_value(value: str) -> str:
        calls.append(value)
        return value

    with pytest.raises(UnexpectedModelBehavior, match=r'(?i)(undeclared|unknown).*tool'):
        await agent.run('use an undeclared tool')
    assert calls == [] and len(sdk.transports) == 1
    assert sdk.transports[0].closed.is_set() and sdk.transports[0].reader_done.is_set()


@pytest.mark.parametrize('structured', [False, True])
async def test_refusal__should_raise_content_filter_without_output_retry(
    model: AnthropicModel, sdk: Harness, structured: bool
) -> None:
    sdk.scenarios = ['refusal']
    with pytest.raises(ContentFilterError):
        if structured:
            await Agent(model, output_type=NativeOutput(Answer), retries=3).run('refuse structured output')
        else:
            await Agent(model).run('refuse text output')
    assert len(sdk.transports) == 1
    assert sdk.transports[0].closed.is_set() and sdk.transports[0].reader_done.is_set()


async def test_future_native_fields__should_not_drop_or_restart_response(model: AnthropicModel, sdk: Harness) -> None:
    sdk.scenarios = ['future-fields']
    result = await Agent(model).run('accept additive native protocol fields')
    assert result.output == 'PRINT_TRANSPORT_OK'
    assert result.usage.input_tokens == 15 and result.usage.output_tokens == 11
    assert len(sdk.transports) == 1 and sdk.transports[0].closed.is_set()


@pytest.mark.parametrize(
    'scenario,error_type',
    [
        ('rate-limit', ClaudeCodeSubscriptionLimitError),
        ('quota', ClaudeCodeUsageCreditsError),
        ('quota-rejected', ClaudeCodeUsageCreditsError),
        ('credit-limit', ClaudeCodeCreditLimitError),
        ('unknown-rate-limit', ModelRateLimitError),
    ],
)
async def test_quota_error__should_preserve_full_public_sdk_data_without_redaction(
    model: AnthropicModel, sdk: Harness, scenario: str, error_type: type[Exception]
) -> None:
    sdk.scenarios = [scenario]
    with pytest.raises(error_type) as caught:
        await Agent(model).run('report subscription limit')
    assert isinstance(caught.value, (ModelQuotaExceededError, ModelRateLimitError))
    body = caught.value.body
    assert isinstance(body, dict)
    diagnostic = json.dumps(body)
    assert 'synthetic-sensitive-limit-diagnostic' in diagnostic
    assert 'unifiedWindows' in diagnostic and 'surpassedThreshold' in diagnostic
    rate_frame = next(frame for frame in frames() if frame['type'] == 'rate_limit_event')
    assert rate_frame['uuid'] in diagnostic and rate_frame['session_id'] in diagnostic
    assert 'headers' not in body  # Do not manufacture unavailable HTTP metadata.
    assert len(sdk.transports) == 1
    assert sdk.transports[0].closed.is_set() and sdk.transports[0].reader_done.is_set()


@pytest.mark.parametrize('scenario', ['quota', 'quota-rejected'])
async def test_usage_credits__should_allow_only_explicit_opt_in(sdk: Harness, scenario: str) -> None:
    sdk.scenarios = [scenario]
    instance = ClaudeCodeModel('haiku', allow_usage_credits=True)
    try:
        result = await Agent(instance).run('explicitly permit usage credits')
        assert result.output == 'PRINT_TRANSPORT_OK'
        assert result.usage.output_tokens == 11
    finally:
        await instance.client.close()
    assert sdk.transports[0].closed.is_set() and sdk.transports[0].reader_done.is_set()


@pytest.mark.parametrize(
    'scenario,error_type',
    [
        ('subscription-status', ClaudeCodeSubscriptionLimitError),
        ('credit-status', ClaudeCodeCreditLimitError),
    ],
)
async def test_billing_error__should_preserve_raw_terminal_payload_and_sdk_cause(
    model: AnthropicModel, sdk: Harness, scenario: str, error_type: type[ClaudeCodeQuotaError]
) -> None:
    sdk.scenarios = [scenario]
    with pytest.raises(error_type) as caught:
        await Agent(model).run('identify the exhausted quota')
    error = caught.value
    transport = sdk.transports[0]
    assert isinstance(error, ModelQuotaExceededError)
    assert isinstance(error.__cause__, ResultError)
    assert error.body is error.__cause__.data
    assert error.body == transport.terminal_result
    assert error.__cause__.__cause__ is transport.process_error
    assert error.rate_limit_info.raw['future_limit_field']['diagnostic'] == 'synthetic-sensitive-limit-diagnostic'
    assert transport.closed.is_set() and transport.reader_done.is_set()


@pytest.mark.parametrize(
    'scenario,error_type',
    [
        ('rate-limit', ClaudeCodeSubscriptionLimitError),
        ('credit-limit', ClaudeCodeCreditLimitError),
    ],
)
async def test_credit_opt_in__should_not_override_actual_subscription_or_credit_rejections(
    sdk: Harness, scenario: str, error_type: type[ClaudeCodeQuotaError]
) -> None:
    sdk.scenarios = [scenario]
    async with ClaudeCodeModel('haiku', allow_usage_credits=True) as instance:
        with pytest.raises(error_type):
            await Agent(instance).run('credits permitted but not available')


@pytest.mark.parametrize('deadline', ['idle', 'total'])
@pytest.mark.parametrize('value', [0.0, -1.0, float('inf'), float('-inf'), float('nan')])
async def test_invalid_timeout__should_reject_before_sdk_client(sdk: Harness, deadline: str, value: float) -> None:
    with pytest.raises(UserError, match='finite positive'):
        if deadline == 'idle':
            ClaudeCodeModel(idle_timeout=value)
        else:
            ClaudeCodeModel(total_timeout=value)
    assert sdk.transports == [] and sdk.preflights == []


@pytest.mark.parametrize(
    'settings',
    [
        AnthropicModelSettings(anthropic_metadata={'user_id': 'application-must-not-own-subscription-attribution'}),
        AnthropicModelSettings(extra_body={'metadata': {'user_id': 'unsupported-api-metadata'}}),
        AnthropicModelSettings(anthropic_cache_instructions=True),
        AnthropicModelSettings(anthropic_cache_instructions='1h'),
        AnthropicModelSettings(max_retries=1),
        AnthropicModelSettings(extra_headers={'Authorization': 'must-not-override-cli-auth'}),
        AnthropicModelSettings(extra_headers={'User-Agent': 'must-not-override-cli-identity'}),
        AnthropicModelSettings(extra_headers={'X-SDK-Test': 'value\nX-API-Key: must-not-inject'}),
    ],
)
async def test_unsupported_settings__should_reject_before_sdk_client(
    model: AnthropicModel, sdk: Harness, settings: AnthropicModelSettings
) -> None:
    with pytest.raises(UserError):
        await model.request(
            [ModelRequest(parts=[SystemPromptPart('instruction'), UserPromptPart('hello')])],
            settings,
            ModelRequestParameters(),
        )
    assert sdk.transports == [] and sdk.preflights == []


async def test_api_tool_flags__should_not_silently_discard_unsupported_strictness(
    model: AnthropicModel, sdk: Harness
) -> None:
    with pytest.raises(UserError, match='strict/cache-control'):
        await model.request(
            [ModelRequest(parts=[UserPromptPart('strict tool')])],
            None,
            ModelRequestParameters(function_tools=[ToolDefinition(name='lookup', strict=True)]),
        )
    assert sdk.preflights == [] and sdk.transports == []


async def test_application_headers__should_use_native_cli_header_setting(model: AnthropicModel, sdk: Harness) -> None:
    await Agent(model, model_settings={'extra_headers': {'X-SDK-Test': 'application-owned'}}).run('hello')
    assert sdk.transports[0].options.env['ANTHROPIC_CUSTOM_HEADERS'] == 'x-sdk-test: application-owned'


@pytest.mark.parametrize(
    'account',
    [
        None,
        {},
        {'apiProvider': 'bedrock', 'subscriptionType': 'Claude Team'},
        {'apiProvider': 'firstParty', 'subscriptionType': None},
        {'apiProvider': 'firstParty', 'subscriptionType': 'API'},
    ],
)
async def test_account_preflight__should_reject_before_sending_any_generation_frames(
    model: AnthropicModel, sdk: Harness, account: dict[str, Any] | None
) -> None:
    sdk.server_info['account'] = account
    with pytest.raises(UserError):
        await Agent(model).run('must never generate on this account')
    assert sdk.transports == [] and len(sdk.preflights) == 1
    transport = sdk.preflights[0]
    assert not transport.started.is_set()
    assert not any(frame['type'] in ('user', 'assistant') for frame in transport.writes)
    assert transport.closed.is_set() and transport.reader_done.is_set()


@pytest.mark.parametrize('executable', [None, '/explicit/claude-override'])
async def test_cli_selection__should_use_bundled_sdk_binary_unless_overridden(
    sdk: Harness, executable: str | None
) -> None:
    instance = ClaudeCodeModel('haiku', sdk_options=ClaudeAgentOptions(cli_path=executable))
    try:
        await Agent(instance).run('select the SDK CLI')
        assert sdk.transports[0].options.cli_path == executable
    finally:
        await instance.client.close()


async def test_model_resolver__should_resolve_public_namespace_without_starting_client(sdk: Harness) -> None:
    context = ModelResolutionContext(agent=Agent(), deps=None)
    assert resolve_model(context, 'other-provider:haiku') is None
    resolved = resolve_model(context, 'claude-code:haiku')
    assert isinstance(resolved, ClaudeCodeModel)
    try:
        assert resolved.model_name == 'haiku' and resolved.system == 'claude-code'
        assert sdk.transports == [] and sdk.preflights == []
    finally:
        await resolved.client.close()
    agent = Agent('claude-code:haiku', capabilities=[ResolveModelId(resolve_model)])
    result = await agent.run('resolve the SDK model')
    assert result.output == 'PRINT_TRANSPORT_OK'
    assert sdk.transports[0].options.model == 'haiku'
    assert sdk.transports[0].options.cli_path is None


@pytest.mark.parametrize('deadline', ['idle', 'total', 'request'])
async def test_timeout__should_close_sdk_and_join_reader(sdk: Harness, deadline: str) -> None:
    sdk.scenarios = ['hang']
    instance = ClaudeCodeModel(
        'haiku', idle_timeout=0.02 if deadline == 'idle' else 2, total_timeout=0.04 if deadline == 'total' else 4
    )
    settings = AnthropicModelSettings(timeout=0.02) if deadline == 'request' else None
    try:
        expected = ModelTimeoutError if deadline == 'request' else TimeoutError
        with anyio.fail_after(5), pytest.raises(expected):
            await instance.request([ModelRequest(parts=[UserPromptPart('wait')])], settings, ModelRequestParameters())
    finally:
        await instance.client.close()
    assert len(sdk.transports) == 1
    assert sdk.transports[0].closed.is_set() and sdk.transports[0].reader_done.is_set()


async def test_request_read_timeout__should_reset_on_progress_not_cap_entire_stream(
    model: AnthropicModel, sdk: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = sdk.query

    async def paced(prompt: AsyncIterable[dict[str, Any]], options: ClaudeAgentOptions) -> AsyncIterator[Message]:
        async for message in original(prompt, options):
            await asyncio.sleep(0.02)
            yield message

    monkeypatch.setattr(sdk_transport, '_query', paced)
    result = await Agent(model, model_settings={'timeout': 0.1}).run('stream with regular progress')
    assert result.output == 'PRINT_TRANSPORT_OK'
    assert sdk.transports[0].closed.is_set() and sdk.transports[0].reader_done.is_set()


@pytest.mark.parametrize('scenario', ['hang', 'tool'])
async def test_repeated_direct_cancellation__should_join_cleanup_before_any_tool_handoff(
    model: AnthropicModel, sdk: Harness, scenario: str
) -> None:
    sdk.scenarios, sdk.block_close = [scenario], True
    calls: list[str] = []
    agent = Agent(model)

    @agent.tool_plain
    def return_value(value: str) -> str:
        calls.append(value)
        return value

    with anyio.fail_after(5):
        async with asyncio.TaskGroup() as group:
            task = group.create_task(agent.run('cancel without handing off a tool'))
            while not sdk.transports:
                await anyio.lowlevel.checkpoint()
            transport = sdk.transports[0]
            try:
                await transport.started.wait()
                if scenario == 'hang':
                    task.cancel()
                await transport.close_started.wait()
                for _ in range(3):
                    task.cancel()
                    await anyio.lowlevel.checkpoint()
                assert not transport.closed.is_set()
                assert not task.done(), 'Cancellation must wait for the SDK close to finish'
                assert calls == []
            finally:
                transport.release_close.set()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert calls == [] and len(sdk.transports) == 1
    assert transport.closed.is_set() and transport.reader_done.is_set()


async def test_configuration__should_copy_env_and_pass_same_context_to_preflight_and_query(
    sdk: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'parent-key-must-be-cleared')
    env = {'APP_CONTEXT': 'configured', 'ANTHROPIC_API_KEY': ''}
    async with ClaudeCodeModel('haiku', sdk_options=ClaudeAgentOptions(cwd=tmp_path, env=env)) as model:
        env['APP_CONTEXT'] = 'later-mutation'
        result = await Agent(model).run('check context')
    assert result.output == 'PRINT_TRANSPORT_OK'
    for transport in [*sdk.preflights, *sdk.transports]:
        assert transport.options.cwd == tmp_path
        assert transport.options.env['APP_CONTEXT'] == 'configured'
        assert transport.options.env['ANTHROPIC_API_KEY'] == ''
        assert transport.closed.is_set() and transport.reader_done.is_set()


@pytest.mark.parametrize(
    'env',
    [
        {'ANTHROPIC_API_KEY': 'not-a-subscription-login'},
        {'ANTHROPIC_BASE_URL': 'https://not-selected.invalid'},
        {'CLAUDE_CODE_MAX_RETRIES': '5'},
        {'ENABLE_TOOL_SEARCH': 'true'},
    ],
)
async def test_environment__should_pass_sdk_owned_options_without_a_local_allowlist(
    sdk: Harness, env: dict[str, str]
) -> None:
    async with ClaudeCodeModel('haiku', sdk_options=ClaudeAgentOptions(env=env)) as model:
        await Agent(model).run('SDK owns its environment')
    for transport in [*sdk.preflights, *sdk.transports]:
        assert all(transport.options.env[name] == value for name, value in env.items())


async def test_sdk_options__should_pass_native_configuration_and_merge_only_request_plumbing(
    sdk: Harness, tmp_path: Path
) -> None:
    async def stop_hook(data: HookInput, tool_id: str | None, context: HookContext) -> HookJSONOutput:
        return {}

    stderr: list[str] = []
    matcher = HookMatcher(hooks=[stop_hook])
    supplied = ClaudeAgentOptions(
        model='claude-haiku-5-5',
        cli_path='/never-launch-claude',
        cwd=tmp_path,
        max_budget_usd=0.25,
        max_turns=4,
        max_buffer_size=1234567,
        thinking={'type': 'adaptive'},
        effort='low',
        output_format={'type': 'json_schema', 'schema': {'type': 'string'}},
        sandbox={'enabled': True},
        add_dirs=[tmp_path / 'extra'],
        user='sdk-user',
        setting_sources=['project'],
        skills=['sample'],
        tools=['Read'],
        disallowed_tools=['WebSearch'],
        include_hook_events=True,
        enable_file_checkpointing=True,
        load_timeout_ms=1234,
        session_store_flush='eager',
        settings='{"caller_setting":true}',
        extra_args={'future-sdk-flag': 'untouched'},
        env={'SDK_APPLICATION': 'test'},
        hooks={'Stop': [matcher]},
        stderr=stderr.append,
        mcp_servers={'caller': {'command': '/never-launch-mcp', 'args': []}},
    )
    async with ClaudeCodeModel(sdk_options=supplied) as instance:
        agent = Agent(instance)

        @agent.tool_plain
        def return_value(value: str) -> str:
            return value

        await agent.run('inspect configuration')
    actual = sdk.transports[0].options
    assert actual.model == supplied.model and actual.cli_path == supplied.cli_path
    assert actual.max_budget_usd == supplied.max_budget_usd and actual.max_turns == supplied.max_turns
    assert actual.max_buffer_size == supplied.max_buffer_size
    assert actual.thinking is supplied.thinking and actual.effort == supplied.effort
    assert actual.output_format is supplied.output_format and actual.sandbox is supplied.sandbox
    assert actual.add_dirs == supplied.add_dirs and actual.user == supplied.user
    assert actual.setting_sources == supplied.setting_sources and actual.skills == supplied.skills
    assert actual.tools is supplied.tools and actual.disallowed_tools is supplied.disallowed_tools
    assert actual.include_hook_events and actual.enable_file_checkpointing
    assert actual.load_timeout_ms == supplied.load_timeout_ms
    assert actual.session_store_flush == supplied.session_store_flush
    assert actual.settings == supplied.settings
    assert actual.extra_args['future-sdk-flag'] == 'untouched'
    assert actual.env['SDK_APPLICATION'] == 'test'
    assert actual.hooks is not None and actual.hooks['Stop'][0] is matcher
    assert 'PreToolUse' in actual.hooks and supplied.hooks == {'Stop': [matcher]}
    assert isinstance(actual.mcp_servers, dict) and isinstance(supplied.mcp_servers, dict)
    assert actual.mcp_servers['caller'] is supplied.mcp_servers['caller']
    assert 'pydantic_ai' in actual.mcp_servers and 'pydantic_ai' not in supplied.mcp_servers
    assert actual.stderr is not None
    actual.stderr('SDK callback data')
    assert stderr == ['SDK callback data']
    assert supplied.env == {'SDK_APPLICATION': 'test'}
    assert supplied.extra_args == {'future-sdk-flag': 'untouched'}


async def test_sdk_model__should_reject_two_conflicting_model_names_before_sdk_io(sdk: Harness) -> None:
    with pytest.raises(UserError, match='must agree'):
        ClaudeCodeModel('haiku', sdk_options=ClaudeAgentOptions(model='sonnet'))
    assert sdk.preflights == [] and sdk.transports == []


@pytest.mark.parametrize('budget', [0, 1024])
async def test_sdk_thinking__should_not_shadow_native_legacy_option_with_api_defaults(
    sdk: Harness, budget: int
) -> None:
    async with ClaudeCodeModel(sdk_options=ClaudeAgentOptions(max_thinking_tokens=budget)) as model:
        await Agent(model).run('SDK controls thinking')
    options = sdk.transports[0].options
    assert options.max_thinking_tokens == budget
    body = json.loads(options.env['CLAUDE_CODE_EXTRA_BODY'])
    assert 'thinking' not in body and 'context_management' not in body
