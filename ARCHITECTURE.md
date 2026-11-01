# Architecture

## One SDK-backed model

`ClaudeCodeModel` subclasses Pydantic AI's `AnthropicModel`. Its Anthropic SDK client has an in-memory HTTPX transport that adapts the prepared request to Anthropic's public Python Agent SDK. It never sends HTTP requests itself. There is no local HTTP listener, bearer-token relay, custom subprocess supervisor, external MCP bridge, or alternate model implementation.

The Agent SDK runs its bundled Claude Code binary by default. Configuration is a single native `ClaudeAgentOptions` object passed as `sdk_options`. There are no mirrored CLI-path, directory, environment, budget, thinking, sandbox, or plugin parameters. SDK fields pass through via `dataclasses.replace`, including future fields and live resource objects. The adapter does not implement OAuth, open credential files, copy authentication material, or provide a login flow. Claude Code owns authentication and talks directly to Anthropic.

Pydantic AI retains its native request mapping, stream decoding, usage accounting, agent loop, Python tool execution, and output validation. The SDK manages the Claude Code process and its control protocol. The adapter adds transport work, not another agent framework.

## One model request

1. Pydantic AI maps its actual history, tool definitions, and settings into a native Anthropic request.
2. A quota-free SDK initialization checks the account before any generation frame is submitted. The successful classification is cached for the model client's lifetime. Both initialization and generation receive caller SDK configuration. SDK authentication/account reports, not an environment-variable allowlist, enforce the subscription requirement.
3. Native user and assistant messages become SDK stream-json frames. Earlier user frames have `shouldQuery: false`. Each must receive a successful zero-turn acknowledgment before replay continues. Tool IDs, arguments, results, thinking signatures, and native content blocks remain structured data.
4. Caller SDK options go directly to the SDK. Without a caller system prompt, the adapter supplies the native preset with Pydantic instructions appended. Two competing system prompts are rejected instead of dropping either. API-only Pydantic settings still use an internal `CLAUDE_CODE_EXTRA_BODY` mapping because the SDK has no generic extra-body property. Explicit SDK thinking, effort, output format, and task-budget options take precedence. SDK-native tool discovery supplies the complete tool catalog; the adapter does not replace it with a partial list. Claude Code constructs its native system blocks, reminders, metadata, and HTTP request.
5. Tool schemas are exposed through an SDK-managed in-process MCP server, merged with caller dictionary-form MCP configuration. Caller hooks and callbacks are retained. An added public `PreToolUse` hook returns `permissionDecision: "defer"` for Pydantic tools. The SDK stops the turn before executing the tool. An inert handler records and rejects any unexpected attempt to execute it. Unsupported API tool strict/cache-control flags are rejected rather than silently lost.
6. Native SDK partial stream events feed the existing Anthropic decoder. Tool names are translated back without touching argument data. Proposals and terminal events are buffered until the SDK has finished and all transport checks have passed.
7. Pydantic AI executes the Python tool or validates the structured output. The next model request replays the resulting Pydantic AI history through a new SDK query.

Defaults select no built-in tools, no filesystem settings, and one SDK turn. Caller options pass through without a local SDK field allowlist. Native partial events are enabled for decoding; Pydantic history frames carry `client_composed: true` so text cannot become a CLI slash command or file expansion. The adapter still represents one native assistant response, not a full autonomous SDK agent or subagent workflow. Neither SDK turn limits nor Pydantic request counts are an independently enforced HTTP-request counter.

The CLI's nonessential-traffic flag suppresses auxiliary SDK inference by default; explicit caller SDK environment configuration wins. There is no copied collection of CLI safety flags. SDK retry, settings-loading, session, and background-request behavior otherwise follows SDK configuration. Opt-in loopback checks isolate user settings and safely ignore local diagnostics requests without forwarding them.

## History and capabilities

History remains in Pydantic AI's message types. The adapter does not flatten the conversation into a user-text checkpoint or require SDK-owned saved sessions. SDK session/history options pass through when explicitly configured; they can change the conversation context and are the caller's responsibility. The adapter does not implement a second session store.

Claude Code's streaming input cannot represent native API `role: system` entries. The model therefore advertises the matching transport capabilities instead of falsely claiming that API role. Pydantic AI's existing fallback preserves changed instruction text at the right conversation boundary and resolves tool availability through current schemas. Native tool-addition and schema-deferral channels are disabled. This uses core mapping, not an adapter-specific history format.

Direct native `role: system` payloads, subscription-metadata overrides, and per-block system-cache options that the SDK cannot preserve are rejected explicitly. The normal agent interface handles instruction and tool changes through the model's declared capabilities.

## Diagnostic fidelity

Preserve everything the SDK exposes. Do not redact provider errors, replace them with generic privacy messages, truncate diagnostics silently, or invent unavailable metadata.

The adapter exhausts public `query()` through process termination. It does not stop at a typed `ResultMessage`: a later SDK `ResultError` can contain the complete raw CLI result, unknown fields, exit status, and the original `ProcessError` cause. Those exceptions are propagated unchanged. A zero-exit failure that supplies only a typed result retains all fields of that SDK object. Stderr callback output is attached to failures.

Rate-limit notifications are retained without immediately aborting the stream. Aborting on a generic rejected-limit event would discard the following detailed terminal error. When a terminal SDK exception exists, it remains primary and the full limit notification is attached. If no terminal exception exists, the explicit SDK limit report raises the corresponding Pydantic AI error.

There is a real SDK boundary, not an adapter redaction. The public SDK does not expose original HTTP response headers, duplicate headers, body bytes, or SSE envelope fields. A loopback HTTP 429 check confirmed that the CLI reports the error status and message but omits arbitrary extra HTTP-body fields. The adapter cannot recreate information the CLI never emits. `provider_details['upstream_http_metadata_available']` is `False`. Exact HTTP-wire diagnostics would require an upstream SDK/CLI feature, not another hidden proxy.

Error payloads and stderr can contain sensitive application data. The application controls their display and storage. Successful doctor reports omit account identifiers, while failure reports retain the supplied SDK diagnostics.

## Cancellation and timeouts

SDK I/O runs in an owned reader task. It is canceled through an AnyIO cancellation scope, allowing the SDK's cleanup shields to finish. Caller `Task.cancel()` is not forwarded into an already-running SDK close. Cleanup is joined even after repeated cancellation. This avoids abandoning suspended SDK generators or releasing a tool while the CLI is still stopping.

Idle, total, and per-request read deadlines are enforced. Bounded input and response buffers raise explicit failures rather than silently truncating data. The SDK owns subprocess termination; the adapter does not claim its own process-group containment or sandbox guarantees.

Cancellation cannot undo inference. Worker retries can repeat inference. Tools with external effects still need application-level idempotency.

## Billing and policy

Reported usage-credit spending fails the request unless explicitly allowed. Subscription-window rejection and paid-credit use are separate SDK signals: a successful response using explicitly permitted credits is not a subscription-limit failure. An SDK-identified credit limit still fails, even with opt-in.

Public `ClaudeCodeSubscriptionLimitError`, `ClaudeCodeCreditLimitError`, and `ClaudeCodeUsageCreditsError` inherit from `ClaudeCodeQuotaError` and core `ModelQuotaExceededError`. SDK metadata, not message matching, selects them. A confirmed terminal HTTP 429 can be classified only when the SDK also identifies the billing limit; the original SDK error remains the cause and its raw payload remains the body. Unknown limits and unrelated SDK failures are not relabeled as subscription exhaustion.

This cannot undo the first charge or prove that no unreported charge occurred. There is no automatic paid retry or API-key fallback, and no invented dollar balance. Token-price estimates are not a subscription invoice.

Using the public SDK does not establish Anthropic approval for this project. OpenCode's reported permission is not transferable. Authentication, billing, and policy remain subject to Anthropic's current terms. This is an independent community extension, not an Anthropic product or an account-safety guarantee.

## Verification

The implementation is checked with Pydantic AI 2.64.0 and Agent SDK 0.2.176. The SDK's bundled CLI is exercised directly by opt-in local tests. No separate CLI installation is needed for those checks.

Offline tests use the real SDK parser, control protocol, hooks, and in-process MCP routing with a public in-memory `Transport`. Socket and subprocess creation are forbidden. Tests cover native history and schemas, streaming, signed thinking, output modes, Python tool ownership, instruction and tool changes, raw SDK terminal errors, authentication preflight, limits, refusals, malformed streams, timeouts, and repeated cancellation.

Opt-in SDK/CLI tests use a literal loopback server returning recorded SSE or a synthetic HTTP error. They cover text, native typed output, tool deferral, complete Python tool loops, ordinary and tool-result replay, changed instructions, and terminal errors. No requests are forwarded to Anthropic for inference.

```sh
make all
PYDANTIC_AI_CLAUDE_CODE_PROBE=I_ACCEPT_LOCAL_CLI_INSPECTION uv run pytest tests/test_sdk_cli.py
```

`tests/test_live.py` is separately gated by `PYDANTIC_AI_CLAUDE_CODE_LIVE=I_ACCEPT_SUBSCRIPTION_USAGE`. It verifies one Python tool call followed by validated output, using at most two subscription-consuming requests. Ordinary tests and CI leave that gate unset.

Shared frame and schema helpers retain reference fixtures executed from `earendil-works/pi-claude-code` at `70abb65186592ca788b29b3db22c3160d629c41f`, under MIT. Those fixtures verify shared conversion rules, not whole-program equivalence with Pi.
