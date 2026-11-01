# Pydantic AI Claude Code

This community extension lets Pydantic AI use your Claude subscription through the public Python Claude Agent SDK.

`ClaudeCodeModel` extends Pydantic AI's `AnthropicModel`. Pydantic AI owns history, Python tools, streaming, and output validation. The Agent SDK manages Claude Code, authentication, and model access. There is one SDK-backed implementation, with no custom HTTP relay or subprocess launcher.

> [!WARNING]
> Subscription access and billing can change. Anthropic's [current guidance](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan) says signed-in Agent SDK usage draws from subscription limits.
>
> That does not establish approval of this adapter. OpenCode's reported permission does not extend to this project. Follow Anthropic's current terms and guidance. This project provides no account-safety guarantee.

## Install

Requires Python 3.11+, Pydantic AI 2.64+, and your own supported Claude subscription login.

```sh
uv add git+https://github.com/h0rv/pydantic-ai-claude-code.git
```

The Agent SDK includes a Claude Code binary, which is used by default. No separate CLI installation is required to run the adapter. An existing Claude Code login is reused. Authentication remains [Claude Code's own login flow](https://code.claude.com/docs/en/authentication); this package does not implement login or read credential files.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai_claude_code import ClaudeCodeModel


async def main():
    async with ClaudeCodeModel('haiku') as model:
        result = await Agent(model).run('Say hello.')
        print(result.output)


asyncio.run(main())
```

Configure Claude Code with the SDK's own type, not a second configuration API:

```python
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    model='haiku',
    cwd='/path/to/project',
    max_budget_usd=0.50,
    effort='low',
)
model = ClaudeCodeModel(sdk_options=options, allow_usage_credits=False)
```

All SDK fields pass through using the SDK dataclass, including settings, sandboxing, plugins, callbacks, environment, and future SDK fields. There is no local field allowlist or mirrored options type. Use `options.cli_path` to select an installed CLI. `doctor(sdk_options=options)` uses the same SDK configuration. Working-directory and environment options affect Claude Code, not Python tools or the parent process.

The adapter merges its MCP server and tool-deferral hook with caller configuration and enables native partial events for Pydantic AI's decoder. It marks its own history frames as composed data. SDK settings strings, paths, callbacks, and resource objects go to the SDK without being reimplemented or deep-copied. The SDK system prompt and Pydantic instructions must not define two competing system prompts. Combining file-form MCP configuration with Pydantic tools requires the SDK's dictionary form.

SDK configuration remains SDK configuration; it does not turn a Pydantic model request into a full SDK agent workflow. Pydantic owns its Python tools and history. The model still requires one native assistant response; SDK-native multi-response agent or subagent workflows are outside that contract.

Some API-level Pydantic settings have no SDK option. Those still need an internal CLI extra-body mapping; SDK-native thinking, effort, output format, and task-budget options take precedence when supplied. Auxiliary SDK inference is suppressed by default through the CLI's environment setting, which callers can override through `options.env`. The adapter does not invent typed SDK fields that the SDK does not provide.

Use standard Pydantic AI tools, output types, settings, and capabilities. Use a full model ID for native schema output or precise thinking settings; CLI aliases can change models. String configuration uses `ResolveModelId(resolve_model)` with the `claude-code:haiku` namespace.

## Behavior and limits

- Python tools execute in Pydantic AI, not in Claude Code. The SDK defers those proposals before execution. Defaults disable built-in tools and filesystem customizations; explicit SDK configuration passes through.
- Native user and assistant history, tool IDs, tool results, and thinking signatures are replayed without converting the conversation into a text checkpoint. Changed instructions and tool availability use Pydantic AI's standard transport-compatible fallback.
- SDK errors, raw terminal error payloads, stderr, and causes are preserved. Upstream HTTP fields that Claude Code does not emit cannot be recovered. No error details are deliberately redacted. See [ARCHITECTURE.md](ARCHITECTURE.md).
- Each model request uses a fresh SDK query. The SDK owns process cleanup. Tools are not handed to Pydantic AI until cleanup finishes.
- Subscription attribution metadata and native system-block caching belong to Claude Code. Incompatible overrides are rejected rather than ignored.
- The SDK owns environment inheritance and configuration validation. Subscription account checks remain in place. API tool strictness and cache overrides that SDK MCP cannot represent raise explicit errors. This is not a sandbox.
- There is no runtime Harness dependency or dedicated CLAI plugin. Harness capabilities compose through the agent's normal `capabilities` argument.

## Subscription and billing

Use your own login. Requests consume your plan's capacity. Cancellation can still consume capacity. Reported usage-credit spending fails the request unless explicitly allowed, but the first charge cannot be undone. Disable usage credits in your account if no credit spending is acceptable.

Catch billing outcomes without parsing error strings:

```python
from pydantic_ai_claude_code import (
    ClaudeCodeCreditLimitError,
    ClaudeCodeSubscriptionLimitError,
    ClaudeCodeUsageCreditsError,
)

try:
    result = await agent.run('Hello')
except ClaudeCodeSubscriptionLimitError as error:
    print('Subscription exhausted:', error.rate_limit_info.rate_limit_type)
    print('Resets at Unix timestamp:', error.rate_limit_info.resets_at)
except ClaudeCodeUsageCreditsError:
    print('Paid usage reported, but this application has not opted in.')
except ClaudeCodeCreditLimitError:
    print('The SDK reported a usage-credit limit.')
```

All three inherit from `ClaudeCodeQuotaError` and Pydantic AI's `ModelQuotaExceededError`. They retain the complete SDK limit event in `error.event`, typed SDK fields in `error.rate_limit_info`, and the original terminal payload and SDK cause when available. Unidentified rate limits remain ordinary rate-limit errors, not guessed subscription failures.

`allow_usage_credits=True` explicitly permits provider-enabled paid usage. It does not enable credits in your account, bypass a credit limit, or automatically retry a failed request. An exhausted subscription window does not reject a successful response served through explicitly permitted credits. Do not retry automatically after asking for consent: the first request may already have spent credits or executed application tools.

The SDK does not provide a reliable remaining dollar balance or a pre-request billing guarantee. Disable paid usage at the account level if charges are unacceptable. The adapter does not silently switch to API-key authentication or another provider.

Token prices reported by Pydantic AI and SDK dollar budgets are not your subscription invoice or a hard account-level spending guarantee. SDK retry and background-request behavior belongs to SDK configuration. Use API keys and supported API models when you need a production support agreement.

## Checks

Quota-free SDK and account diagnostics:

```sh
uv run python -m pydantic_ai_claude_code
```

Offline tests and quality checks:

```sh
make all
uv build
```

Real SDK and bundled CLI against a local mock server, without inference:

```sh
PYDANTIC_AI_CLAUDE_CODE_PROBE=I_ACCEPT_LOCAL_CLI_INSPECTION uv run pytest tests/test_sdk_cli.py
```

An explicitly opted-in test of Python tools and typed output uses at most two subscription requests:

```sh
PYDANTIC_AI_CLAUDE_CODE_LIVE=I_ACCEPT_SUBSCRIPTION_USAGE uv run pytest tests/test_live.py -s
```

## Attribution

Native frame and schema preparation was adapted from the Pi team's [pi-claude-code](https://github.com/earendil-works/pi-claude-code), under its MIT license. The previous custom relay has been removed. The model-access integration now uses Anthropic's public [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview).
