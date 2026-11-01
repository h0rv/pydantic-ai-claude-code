"""Opt-in real SDK/bundled-CLI checks against a local mock, without inference."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import h11
import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    Message,
    ResultError,
    ResultMessage,
    query as sdk_query,
)
from pydantic import BaseModel, JsonValue, TypeAdapter
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, override_allow_model_requests
from pydantic_ai.tools import ToolDefinition

import pydantic_ai_claude_code._transport as sdk_transport
from pydantic_ai_claude_code import ClaudeCodeModel

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        os.environ.get('PYDANTIC_AI_CLAUDE_CODE_PROBE') != 'I_ACCEPT_LOCAL_CLI_INSPECTION',
        reason='Explicit opt-in required for quota-free SDK/CLI loopback checks.',
    ),
]
OBJECT = TypeAdapter(dict[str, JsonValue])
FIXTURES = Path(__file__).parent / 'fixtures'


@dataclass
class Wire:
    calls: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    sessions: list[ClaudeSDKClient] = field(default_factory=list[ClaudeSDKClient])
    results: list[ResultMessage] = field(default_factory=list[ResultMessage])
    custom_headers: list[str] = field(default_factory=list[str])


class NativeAnswer(BaseModel):
    value: str


def events(scenario: str, request_index: int) -> Iterator[dict[str, JsonValue]]:
    tool = scenario == 'tool' or (scenario == 'tool-loop' and request_index == 0)
    fixture = FIXTURES / ('tool-stream.jsonl' if tool else 'text-stream.jsonl')
    first_text = True
    for line in fixture.read_bytes().splitlines():
        record = OBJECT.validate_json(line)
        if record['type'] != 'stream_event':
            continue
        value = OBJECT.validate_python(record['event'], strict=True)
        if tool and value['type'] == 'content_block_start':
            block = OBJECT.validate_python(value['content_block'], strict=True)
            if block['type'] == 'tool_use':
                block['name'] = 'mcp__pydantic_ai__lookup'
                value['content_block'] = block
        if scenario == 'native' and value['type'] == 'content_block_delta':
            value['delta'] = {'type': 'text_delta', 'text': '{"value":"SDK_NATIVE_OK"}' if first_text else ''}
            first_text = False
        yield value


async def send_response(conn: h11.Connection, writer: asyncio.StreamWriter, scenario: str, request_index: int) -> None:
    if scenario == 'error':
        payload = json.dumps(
            {
                'type': 'error',
                'error': {
                    'type': 'rate_limit_error',
                    'message': 'synthetic SDK limit diagnostic',
                    'extra': {'unknown': 'unredacted'},
                },
                'request_id': 'req-local-only',
            }
        ).encode()
        writer.write(
            conn.send(
                h11.Response(
                    status_code=429,
                    headers=[(b'content-type', b'application/json'), (b'request-id', b'req-local-only')],
                )
            )
        )
        writer.write(conn.send(h11.Data(data=payload)))
    else:
        writer.write(conn.send(h11.Response(status_code=200, headers=[(b'content-type', b'text/event-stream')])))
        for value in events(scenario, request_index):
            writer.write(conn.send(h11.Data(data=f'event: {value["type"]}\ndata: {json.dumps(value)}\n\n'.encode())))
    writer.write(conn.send(h11.EndOfMessage()))
    await writer.drain()


async def recorded_query(
    prompt: AsyncIterable[dict[str, JsonValue]], options: ClaudeAgentOptions, url: str, wire: Wire
) -> AsyncIterator[Message]:
    options.env['ANTHROPIC_BASE_URL'] = url
    async for message in sdk_query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            wire.results.append(message)
        yield message


@pytest.fixture
async def wire(monkeypatch: pytest.MonkeyPatch, scenario: str) -> AsyncIterator[Wire]:
    state = Wire()
    tasks: set[asyncio.Task[None]] = set()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = h11.Connection(h11.SERVER)
        payload = bytearray()
        model_request = False
        try:
            while True:
                event = conn.next_event()
                if event is h11.NEED_DATA:
                    chunk = await reader.read(65536)
                    if not chunk:
                        return
                    conn.receive_data(chunk)
                elif isinstance(event, h11.Request):
                    model_request = event.target.split(b'?', 1)[0] == b'/v1/messages'
                    # Never capture authentication headers, even in local probes.
                    if model_request:
                        state.custom_headers.extend(
                            value.decode('ascii') for name, value in event.headers if name == b'x-sdk-parity'
                        )
                elif isinstance(event, h11.Data):
                    payload.extend(event.data)
                elif isinstance(event, h11.EndOfMessage):
                    break
            if not model_request:
                # SDK diagnostics/telemetry are not model requests. Do not parse
                # or store their contents; nothing is forwarded upstream.
                writer.write(conn.send(h11.Response(status_code=204, headers=[])))
                writer.write(conn.send(h11.EndOfMessage()))
                await writer.drain()
                return
            state.calls.append(OBJECT.validate_json(bytes(payload)))
            await send_response(conn, writer, scenario, len(state.calls) - 1)
        finally:
            writer.close()
            await writer.wait_closed()

    async with asyncio.TaskGroup() as group:

        def connected(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = group.create_task(serve(reader, writer))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        server = await asyncio.start_server(connected, '127.0.0.1', 0)
        url = f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}'

        def client(options: ClaudeAgentOptions) -> ClaudeSDKClient:
            options.env['ANTHROPIC_BASE_URL'] = url
            session = ClaudeSDKClient(options)
            state.sessions.append(session)
            return session

        def query(prompt: AsyncIterable[dict[str, JsonValue]], options: ClaudeAgentOptions) -> AsyncIterator[Message]:
            # Both public SDK entry points are forced to the literal-loopback mock.
            return recorded_query(prompt, options, url, state)

        monkeypatch.setattr(sdk_transport, '_client', client)
        monkeypatch.setattr(sdk_transport, '_query', query)
        try:
            with override_allow_model_requests(True):
                yield state
        finally:
            server.close()
            await server.wait_closed()
            for task in tuple(tasks):
                task.cancel()


def history_for(scenario: str) -> list[ModelMessage]:
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('Local SDK inspection only.')])]
    if scenario in ('history', 'instruction-history'):
        history.extend(
            [ModelResponse(parts=[TextPart('Previous answer.')]), ModelRequest(parts=[UserPromptPart('Continue.')])]
        )
    if scenario == 'instruction-history':
        history[-1] = ModelRequest(
            parts=[SystemPromptPart('Keep this changed instruction.'), UserPromptPart('Continue.')]
        )
    if scenario == 'tool-history':
        history.extend(
            [
                ModelResponse(parts=[ToolCallPart('lookup', {'value': 'prior'}, tool_call_id='toolu_prior')]),
                ModelRequest(parts=[ToolReturnPart('lookup', 'prior result', tool_call_id='toolu_prior')]),
            ]
        )
    return history


async def run_model(model: ClaudeCodeModel, scenario: str) -> ModelResponse:
    if scenario == 'native':
        agent = Agent(model, output_type=NativeOutput(NativeAnswer), model_settings={'max_tokens': 64})
        result = await agent.run('Return the offline structured response.')
        assert result.output == NativeAnswer(value='SDK_NATIVE_OK')
        return result.response
    if scenario == 'tool-loop':
        calls: list[str] = []
        agent = Agent(model, retries=0)

        @agent.tool_plain
        def lookup(value: str) -> str:
            calls.append(value)
            return 'PYTHON:' + value

        result = await agent.run('Use lookup and then answer.')
        assert calls == ['MCP_BRIDGE_OK'] and result.usage.requests == 2
        return result.response
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='lookup')] if scenario in ('tool', 'tool-history') else []
    )
    prepared = model.prepare_messages(history_for(scenario), parameters)
    return await model.request(prepared, {'max_tokens': 64}, parameters)


@pytest.mark.parametrize(
    'scenario', ['text', 'tool', 'history', 'native', 'tool-history', 'instruction-history', 'tool-loop', 'error']
)
async def test_sdk_cli__should_preserve_agent_behavior_without_upstream_inference(
    wire: Wire, scenario: str, tmp_path: Path
) -> None:
    async with ClaudeCodeModel(
        'claude-haiku-5-5',
        sdk_options=ClaudeAgentOptions(
            cwd=tmp_path,
            env={'ANTHROPIC_API_KEY': '', 'CLAUDE_CODE_MAX_RETRIES': '0'},
            setting_sources=[],
            strict_mcp_config=True,
            skills=[],
            tools=[],
            extra_args={'prompt-suggestions': 'false', 'no-session-persistence': None},
        ),
        total_timeout=30,
        settings={'extra_headers': {'X-SDK-Parity': 'application-owned'}},
    ) as model:
        if scenario == 'error':
            with pytest.raises(ResultError) as caught:
                await run_model(model, scenario)
            assert caught.value.api_error_status == 429
            diagnostic = json.dumps(caught.value.data)
            assert 'synthetic SDK limit diagnostic' in diagnostic
            # The SDK preserves the terminal CLI result. The CLI does not
            # promise arbitrary HTTP body fields or response headers here.
            assert caught.value.data['is_error'] is True
            assert len(wire.calls) == 1
            assert wire.custom_headers == ['application-owned']
            return
        response = await run_model(model, scenario)
    if scenario == 'tool':
        assert isinstance(response.parts[-1], ToolCallPart) and response.parts[-1].tool_name == 'lookup'
        assert wire.results[-1].deferred_tool_use is not None
        assert wire.results[-1].deferred_tool_use.name == 'mcp__pydantic_ai__lookup'
    elif scenario != 'native':
        assert response.parts == [TextPart('PRINT_TRANSPORT_OK')]
    assert len(wire.calls) == (2 if scenario == 'tool-loop' else 1)
    assert len(wire.sessions) == 1
    assert wire.custom_headers == ['application-owned'] * len(wire.calls)
    expected_content = {
        'instruction-history': 'Keep this changed instruction.',
        'tool-history': 'toolu_prior',
        'tool-loop': 'PYTHON:MCP_BRIDGE_OK',
    }
    if scenario in expected_content:
        assert expected_content[scenario] in json.dumps(wire.calls[-1]['messages'])
    assert wire.calls[0]['metadata'] and wire.calls[0]['system']
