"""Shared native request preparation. No process, network, or credential access."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import replace

from pydantic import JsonValue, TypeAdapter
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.tools import ToolDefinition

_OBJECT = TypeAdapter(dict[str, JsonValue])
_OBJECTS = TypeAdapter(list[dict[str, JsonValue]])
CLI_HEADERS = frozenset(
    {
        'authorization',
        'x-api-key',
        'anthropic-version',
        'host',
        'user-agent',
        'content-type',
        'content-length',
        'accept',
        'accept-encoding',
        'connection',
        'x-app',
        'x-claude-code-session-id',
    }
)


def application_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for name, value in headers:
        if name.lower() in CLI_HEADERS or name.lower().startswith('x-stainless-'):
            raise UserError('Claude Code owns this authentication/protocol header: ' + name)
        if any(character in name + value for character in ('\r', '\n')):
            raise UserError('HTTP headers cannot contain carriage returns or newlines.')
        result.append((name, value))
    return result


def object_value(value: JsonValue) -> dict[str, JsonValue]:
    return _OBJECT.validate_python(value, strict=True)


def text_value(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise UnexpectedModelBehavior('Expected a protocol string.')
    return value


def catalog(tools: list[ToolDefinition]) -> tuple[list[ToolDefinition], dict[str, str]]:
    catalog: list[ToolDefinition] = []
    names: dict[str, str] = {}
    for tool in sorted(tools, key=lambda item: item.name):
        alias = (
            tool.name
            if re.fullmatch(r'[a-zA-Z0-9_-]{1,48}', tool.name)
            else 'tool_' + hashlib.sha256(tool.name.encode()).hexdigest()[:32]
        )
        transport_name = 'mcp__pydantic_ai__' + alias
        if transport_name in names:
            raise UserError('Tool names collide after MCP normalization.')
        names[transport_name] = tool.name
        catalog.append(replace(tool, name=alias))
    return catalog, names


def tool_blocks(content: JsonValue) -> Iterator[dict[str, JsonValue]]:
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get('type')
        if kind in ('tool_use', 'tool_reference'):
            yield block
        elif kind in ('tool_addition', 'tool_removal'):
            yield from tool_blocks([block.get('tool')])
        elif kind == 'tool_result':
            yield from tool_blocks(block.get('content'))


def tool_name_key(block: dict[str, JsonValue]) -> str:
    return 'tool_name' if block.get('type') == 'tool_reference' and 'tool_name' in block else 'name'


def prepare(body: dict[str, JsonValue]) -> tuple[dict[str, JsonValue], list[ToolDefinition], dict[str, str]]:
    definitions: list[ToolDefinition] = []
    tools = _OBJECTS.validate_python(body.get('tools', []), strict=True)
    for tool in tools:
        if tool.get('type', 'custom') != 'custom':
            raise UserError('Claude Code relay does not support provider-native tools.')
        definitions.append(
            ToolDefinition(
                name=text_value(tool.get('name')),
                description=text_value(tool.get('description', '')),
                parameters_json_schema=object_value(tool.get('input_schema')),
            )
        )
    manifest, names = catalog(definitions)
    historical: set[str] = set()
    messages = _OBJECTS.validate_python(body.get('messages'), strict=True)
    for message in messages:
        for block in tool_blocks(message.get('content')):
            historical.add(text_value(block.get(tool_name_key(block))))
    _, all_names = catalog([ToolDefinition(name=name) for name in historical | {d.name for d in definitions}])
    forward = {original: alias for alias, original in all_names.items()}
    for tool in tools:
        tool['name'] = forward[text_value(tool.get('name'))]
    for message in messages:
        for block in tool_blocks(message.get('content')):
            key = tool_name_key(block)
            block[key] = forward[text_value(block.get(key))]
    result: dict[str, JsonValue] = {**body, 'messages': [*messages], 'stream': True}
    if 'tools' in body:
        result['tools'] = [*tools]
    choice = result.get('tool_choice')
    if isinstance(choice, dict) and choice.get('type') == 'tool':
        choice['name'] = forward[text_value(choice.get('name'))]
    return result, manifest, names


def system_blocks(value: JsonValue) -> list[dict[str, JsonValue]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{'type': 'text', 'text': value}]
    return _OBJECTS.validate_python(value, strict=True)


def extra_body(desired: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Supply application API options before the CLI constructs its native body."""
    if desired.get('metadata'):
        raise UserError('Claude Code owns request metadata for subscription attribution; it cannot be overridden.')
    body = {key: value for key, value in desired.items() if key not in {'model', 'messages', 'system', 'metadata'}}
    body.setdefault('tools', [])
    body.setdefault('thinking', {'type': 'disabled'})
    if object_value(body['thinking']).get('type') == 'disabled':
        body.setdefault('context_management', {'edits': []})
    return body


def input_frames(desired: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    """Replay the actual canonical history, using Pi's shouldQuery boundary."""
    frames: list[dict[str, JsonValue]] = []
    for message in _OBJECTS.validate_python(desired.get('messages'), strict=True):
        role = text_value(message.get('role'))
        if role == 'system':
            continue
        if role not in ('user', 'assistant'):
            raise UserError('Claude Code history must contain native user, assistant, or system messages.')
        content = message.get('content')
        blocks: JsonValue = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
        if not isinstance(blocks, list):
            raise UserError('Claude Code message content must be text or native content blocks.')
        if not blocks:
            continue
        if role == 'user' and frames and frames[-1]['type'] == 'user':
            previous = object_value(frames[-1]['message'])
            previous_blocks = previous['content']
            assert isinstance(previous_blocks, list)
            previous_blocks.extend(blocks)
            frames[-1]['message'] = previous
        else:
            frames.append({'type': role, 'message': {**message, 'content': blocks}})
    if not frames or frames[-1]['type'] != 'user':
        raise UserError('Claude Code history must end in a non-empty user or tool-result message.')
    for frame in frames[:-1]:
        if frame['type'] == 'user':
            frame['shouldQuery'] = False
    return frames


def system_prompt(desired: dict[str, JsonValue]) -> str:
    blocks = system_blocks(desired.get('system'))
    if any(set(block) - {'type', 'text'} for block in blocks):
        raise UserError('Claude Code owns native system-block caching; per-block system options are unsupported.')
    return '\n'.join(text_value(block.get('text')) for block in blocks)
