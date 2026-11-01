"""Offline comparisons with outputs executed from the pinned Pi extension.

Fixture source: earendil-works/pi-claude-code at
70abb65186592ca788b29b3db22c3160d629c41f, executed with Bun and pi-ai 1.4.0.
No Claude executable, account, or network is used by these tests.
"""

from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter
from pydantic_ai.exceptions import UserError

from pydantic_ai_claude_code._request import (
    extra_body as _extra_body,
    input_frames as _input_frames,
    system_prompt as _system_prompt,
)

OBJECT = TypeAdapter(dict[str, JsonValue])
OBJECTS = TypeAdapter(list[dict[str, JsonValue]])
REFERENCE = OBJECT.validate_json((Path(__file__).parent / 'fixtures' / 'pi-parity.json').read_bytes())


def obj(value: JsonValue) -> dict[str, JsonValue]:
    return OBJECT.validate_python(value, strict=True)


@pytest.mark.parametrize('case', OBJECTS.validate_python(REFERENCE['buildExtraBody'], strict=True))
def test_extra_body__should_match_executed_pi_reference(case: dict[str, JsonValue]) -> None:
    source = obj(case['input'])
    options = obj(source.get('options', {}))
    desired: dict[str, JsonValue] = {'tools': source['tools']}
    if 'maxTokens' in options:
        desired['max_tokens'] = options['maxTokens']
    if options.get('reasoning'):
        desired['thinking'] = {'type': 'adaptive'}
        desired['output_config'] = {'effort': 'low' if options['reasoning'] == 'minimal' else options['reasoning']}
    assert _extra_body(desired) == case['output']


def test_input_frames__should_match_executed_pi_reference() -> None:
    expected = OBJECTS.validate_python(obj(REFERENCE['prepareFrames'])['output'], strict=True)
    final = obj(expected[-1]['message'])
    final_content = TypeAdapter(list[JsonValue]).validate_python(final['content'], strict=True)
    desired: dict[str, JsonValue] = {
        'messages': [
            {'role': 'user', 'content': 'Inspect key'},
            expected[1]['message'],
            {'role': 'user', 'content': [final_content[0]]},
            {'role': 'user', 'content': [final_content[1]]},
        ]
    }
    assert _input_frames(desired) == expected


def test_extra_body__should_preserve_native_options_without_replacing_cli_context() -> None:
    desired: dict[str, JsonValue] = {
        'model': 'haiku',
        'system': 'Application instruction',
        'messages': [{'role': 'user', 'content': 'Application question'}],
        'output_config': {'format': {'type': 'json_schema', 'schema': {'type': 'object'}}},
        'unknown_application_option': {'preserve': [1, 2, 3]},
    }
    extra = _extra_body(desired)
    assert not {'model', 'system', 'messages', 'metadata'} & extra.keys()
    assert extra['unknown_application_option'] == desired['unknown_application_option']
    assert extra['output_config'] == desired['output_config']
    with pytest.raises(UserError, match='subscription attribution'):
        _extra_body({'metadata': {'user_id': 'Different identity'}})


def test_system_prompt__should_use_cli_construction_without_silent_cache_option_loss() -> None:
    assert _system_prompt({'system': 'Application instruction'}) == 'Application instruction'
    assert (
        _system_prompt({'system': [{'type': 'text', 'text': 'First'}, {'type': 'text', 'text': 'Second'}]})
        == 'First\nSecond'
    )
    with pytest.raises(UserError, match='system-block caching'):
        _system_prompt({'system': [{'type': 'text', 'text': 'First', 'cache_control': {'type': 'ephemeral'}}]})
