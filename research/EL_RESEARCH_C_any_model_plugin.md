# EL Research C: packaging the verification and decision layer for any model

Research only; no project files changed. Tags: **[F]** documented fact (source cited), **[S]** secondary or unverified source, **[I]** my inference.

## 1. Existing integration standards

- **MCP** [F]: JSON-RPC 2.0 between host, client and server. Servers expose tools, resources and prompts over stdio or Streamable HTTP. Hosts must get explicit user consent before invoking any tool, and tool annotations are untrusted unless the server is trusted [1]. The architecture page documents a newer stateless revision, 2026-07-28 (`server/discover`, per-request `_meta`, sampling deprecated) [2]; the spec page I read names schema 2025-11-25 [1]. Pin the revision before depending on it. Maturity: high, multi-vendor.
- **ACP** [F]: Zed's editor-to-agent protocol, modelled on LSP. Local agents run as editor subprocesses over JSON-RPC on stdio; remote HTTP/WebSocket is still in progress [3]. Has v1 and v2 specs, including a v2 permission-requests RFD [4]. Claude Code and Codex reach ACP editors through adapters [5, S]. Maturity: medium, editor-focused.
- **OpenAI function calling** [F, Responses API]: tools with name, description, JSON-Schema parameters and optional `strict`; `tool_choice` auto/required/forced/none; results return as `function_call_output` keyed by `call_id` [6]. Chat Completions uses `tools[]`, `tool_calls` and `role: "tool"` with `tool_call_id` [7, S]. Most local servers copy this shape [7, S]. De facto standard.
- **Anthropic Messages** [F]: tools with `input_schema`; `stop_reason: "tool_use"` with `tool_use` blocks; results sent back as `tool_result` with `tool_use_id`; `tool_choice` auto/any/tool; the API is stateless, so full history is resent [8].
- **LiteLLM** [F]: OpenAI-compatible proxy on port 4000; `config.yaml` maps a `model_name` alias to provider params; several deployments can share one alias [9]. Tool-call passthrough is not confirmed by that page. Fallbacks and budgets are documented only in secondary sources [10, S].
- **Claude Code hooks** [F]: 33 events including PreToolUse, PostToolUse and PermissionRequest. PreToolUse returns `permissionDecision` allow/deny/ask/defer and can rewrite arguments via `updatedInput`. Exit 2 blocks the call, and exit 2 still blocks even if stdout says allow. PostToolUse runs after the tool, so it cannot stop execution [11]. Plugin bundling of hooks is from a search snippet [12, S].
- **Aider** [F]: `--lint-cmd` and `--test-cmd`; linting runs after edits by default; `--auto-test` runs tests after each edit; a non-zero exit triggers a fix attempt [13].
- **Cline** [S]: MCP servers configured in `~/.cline/mcp.json` as STDIO or remote HTTP/SSE; tool calls need user approval unless auto-approve is on [14].
- **Continue** [S]: custom models via `provider: openai` plus `apiBase`; a `context` list of providers [15]. Taken from a mirror, not the official site.

**Key point [I]:** MCP and ACP define how tools are exposed, not who must approve or execute them. Hosts run their own built-in Read/Edit/Bash. A pipeline can only gate those by sitting in the execution path: a hook, the model's wire traffic, or its own agent loop.

## 2. Three packagings

**A. MCP server** (`propose_filter`, `verify`, `rank` as tools)
- Host must support MCP tools (stdio or Streamable HTTP) and per-call user approval.
- Pros: broad host support; no model access needed; JSON-Schema tool definitions.
- Cons: the host's model and its built-in Edit/Bash bypass the controller. The server can advise but cannot veto host actions [I].
- Hidden states: none. Head disabled; ranking uses trusted evidence only. Degrades cleanly.
- Journal and sandbox: owned by the server process; journal exposed as a read-only resource.
- Auth: stdio trusts the local user process; Streamable HTTP needs OAuth or bearer tokens [2].

**B. OpenAI-compatible proxy** (wraps `/v1/chat/completions` or `/v1/responses`)
- Host must allow a base-URL override (Continue `apiBase` [15]; most OpenAI-SDK clients). Anthropic-format hosts need a second adapter, since the wire format differs [8].
- Pros: sees every proposed tool call before the host runs it, so it can filter and dry-run in-line regardless of host. The only packaging that gates the model's own proposals without host cooperation [I]. Can sit in front of any upstream, including LiteLLM [9].
- Cons: must relay streaming tool-call deltas, which arrive by `index` [7, S]; must pass reasoning items back for reasoning models [6]; maps host-executed tools to our Action schema; two wire formats to maintain.
- Hidden states: only when the proxy also hosts the model. Remote API: `has_hidden_states=false`, head skipped.
- Journal and sandbox: proxy writes the journal; sandbox runs locally.
- Auth: bind 127.0.0.1, require a local token, keep upstream keys server-side and out of the journal [I].

**C. Python library with a model adapter** (controller runs the loop)
- Host must do nothing; a CLI, script or thin wrapper drives it.
- Pros: full control of the commit point; hidden states available for local models; easiest to test deterministically; no wire-format issues.
- Cons: does not plug into existing agents; each host needs a shim.
- Hidden states: returned when `capabilities.has_hidden_states`; otherwise the scorer is off.
- Journal and sandbox: owned by the library; keys from environment, never in the journal.

## 3. Minimal model adapter interface (proposal)

```python
@dataclass(frozen=True)
class Capabilities:
    has_hidden_states: bool
    supports_tools: bool            # native tool-call format available
    max_context: int
    thinking_control: Literal["none", "budget", "toggle"]
    parallel_tool_calls: bool
    streaming: bool

class ModelAdapter(Protocol):
    capabilities: Capabilities
    def propose(self, req: ProposeRequest) -> ProposeResult: ...
    def hidden_states(self, req: ProposeRequest) -> HiddenStates | None: ...  # only if has_hidden_states
    def format_observation(self, result: ActionResult, call_id: str) -> dict: ...
    def close(self) -> None: ...
```

- `ProposeRequest`: neutral messages, our Read/Edit/Bash JSON schemas, `n_candidates`, `max_tokens`, `thinking_budget`.
- `ProposeResult`: `candidates: list[CandidateAction{tool, args, raw_text, call_id}]`, `usage`, `model_revision` (hash, written to the journal).
- `HiddenStates`: per-candidate vectors, layer id, model revision hash.
- Each adapter maps its wire format to `CandidateAction`: Responses `function_call`, Chat `tool_calls`, Anthropic `tool_use`.

[I] Keep it this small. Add methods only when a second host needs them.

## 4. Recommendation and staged plan

1. **Stage 0 (first):** freeze the adapter interface and the neutral Action schema. Build a fake adapter so filters, dry-run, protected tests and the hash-chained journal are tested with no model. Gate: deterministic tests pass.
2. **Stage 1:** the library (C) plus one adapter for OpenAI-compatible chat/responses. This covers local servers, LiteLLM and most APIs, and is the only route that enforces commits on any model.
3. **Stage 2:** a Claude Code PreToolUse hook as a host-specific gate. It denies Edit/Bash unless the controller has authorized the action, with a reason string [11]. Real enforcement, but Claude Code only.
4. **Stage 3:** the proxy (B) for hosts that accept a base URL, then an Anthropic-format adapter [8].
5. **Defer:** the MCP server (A). It cannot gate host-native tools, so it is useful only as advisory `verify` and journal-read for hosts with no hook. Revisit once the MCP revision is pinned [2].
6. **Defer:** an ACP adapter until remote transport is settled [3].

**Degradation [I]:** for API-only models, deterministic filters, protected tests, trusted-evidence ranking and the journal all still run. Only the learned scorer is lost, and it stays local-only.

**Unverified:** ACP method names (session/new, session/prompt) come from a secondary source; LiteLLM tool passthrough; Continue's context-provider schema; Chat Completions shape from a vLLM example rather than OpenAI's own reference.

## Sources

1. https://modelcontextprotocol.io/specification/2025-11-25 (MCP spec: transports, primitives, security principles)
2. https://modelcontextprotocol.io/docs/learn/architecture (MCP architecture; 2026-07-28 revision)
3. https://agentclientprotocol.com/ (ACP overview, transports)
4. https://agentclientprotocol.com/llms.txt (ACP page index)
5. https://agentic-ai.readthedocs.io/en/latest/Standards/agent-client-protocol/ (secondary, ACP adapters)
6. https://developers.openai.com/api/docs/guides/function-calling (OpenAI function calling, Responses API)
7. https://docs.vllm.ai/en/v0.8.5/getting_started/examples/openai_chat_completion_client_with_tools.html (secondary, Chat Completions tool shape)
8. https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview (Anthropic tool use)
9. https://docs.litellm.ai/docs/proxy/quick_start (LiteLLM proxy)
10. https://www.morphllm.com/llm-proxy (secondary, LiteLLM routing and budgets)
11. https://code.claude.com/docs/en/hooks (Claude Code hooks)
12. https://docs.claude.com/en/docs/claude-code/plugins-reference (search snippet only)
13. https://aider.chat/docs/usage/lint-test.html (Aider lint and test)
14. https://docs.cline.bot/mcp/mcp-overview (Cline MCP; search snippet)
15. https://docs.continue.dev/reference (Continue config; mirror, unverified)
