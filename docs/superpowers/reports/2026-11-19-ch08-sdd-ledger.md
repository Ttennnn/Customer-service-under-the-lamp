# SDD ledger — plan: docs/superpowers/plans/2026-11-19-ecommerce-cs-ch08-tool-system.md

Spec: docs/superpowers/specs/2026-11-19-ecommerce-cs-ch08-tool-system-design.md
Branch: feature/ch08-tool-system
Setup commit: 9e7afac (design + plan + dev-notes)
Task 1 BASE: 9e7afac

## Pre-flight scan

| Pair / task | Produces vs consumes | Finding / ruling |
|---|---|---|
| Task 2 → 3/4/5/6/8 | `spec_from_tool`, `builtin_specs`, `get_all_specs`, `ToolSpec` | consistent; Task 2 is the single source |
| Task 2 → 7 | `get_all_specs` needs `fetch_mcp_specs`; mcp_client is Task 7 | Ruling: Task 2 defines `fetch_mcp_specs(settings) -> []` as a temporary stub; Task 7 replaces it with the real client call. Cost if wrong: one extra edit in Task 7. |
| Task 3 ↔ 4 | both add `SPECS` to `app/tools/builtin/`; task 3 asserts incremental names | no conflict; assertions split incrementally |
| Task 3 → graph | `fetch_order_node` currently uses `execute_tool(query_order)`; Task 3 removes it | Ruling: Task 3 changes `fetch_order_node` to `order_snapshot(order_id)` so it stops using the retired executor path. Cost if wrong: policy lookup test needs a small adjustment. |
| Task 6 → 7 | engine's execution loop needs MCP async path; async_bridge is Task 7 | Ruling: Task 6 implements sync execution only; Task 7 adds the MCP branch to `engine.execute`. Cost if wrong: Task 7 touches one function it already owns. |
| Task 6 → 9 | `execute`, `deny_validation`, `deny_permission`, `ToolRun` | consistent signatures |
| Task 8 ↔ 9 | Task 8 binds `create_ticket` but has no interrupt; Task 9 adds it | Ruling: accept the intermediate state where an unconfirmed write is permission-denied. Task 9 replaces it with the interrupt flow. Cost if wrong: none, intermediate commit only. |
| Task 9 ↔ 10 | both touch `service.py`; Task 10 adds ticket-number fallback | sequential; Task 9 leaves `created_ticket_no` in State, Task 10 consumes |
| Task 9 ↔ 13 | interrupt frame contract `kind=confirm_ticket`, `preview` | consistent |
| Task 10 ↔ ch05/ch06 | both endpoints consume `ToolRun.data` | consistent |
| Task 11 ↔ 6 | legacy chat consumes `execute_tool_call` | consistent |
| Task 12 | intent only; no shared file with 1–11 except `state.py` | `Intent` literal addition; no conflict |
| Task 14 | consumes all | no conflict |
| Per-task self-check | Task 2 test monkeypatches `app.tools.registry.fetch_mcp_specs` | covered by the Task 2 stub ruling |
| Per-task self-check | Task 3 expected builtin list | fixed to incremental assertion in plan |
| Per-task self-check | Task 9 dual-tool test needs one AIMessage with two tool calls | fixed in plan with `_dual_tool_chunk` |
| Task 2 ↔ existing consumers | Rewriting `registry.py` would remove `ALL_TOOLS` / `TOOLS_BY_NAME` / `TOOL_LABELS` / `tool_timeout` while `main.py`, `services/chat.py`, `agent/subgraph.py` still import them | Ruling: Task 2 adds the new registry API alongside the legacy exports; legacy exports stay until Tasks 3/4/8/11 migrate their consumers. Cost if wrong: one compatibility cleanup later. |

## Progress

- [x] Task 1: complete (commits 9e7afac..4113682, review clean; 2 Minor notes parked: ORM/DDL width drift, thin model test)
- [x] Task 2: complete (commits 4113682..548e747, review clean; Minor notes parked: partial scan retry, silent no-SPECS modules, register source ambiguity, shallow schema test, no legacy-export test, test global-state reset, settings=None branch, unhashable frozen spec)
- [x] Task 3: complete (commits 548e747..730b588, review clean; Minor notes parked: empty order_id returns null — Task 5 formatter must tolerate None; determinism comment; stale test names/fixtures; prompt advertises MCP tool before Task 7)
- [x] Task 4: complete (commits 730b588..027fb39, review clean; Minor notes parked: logger namespace, global time.sleep patch, legacy query_faq timeout 25 vs spec 30 to reconcile in Task 11, exact-set test brittleness)
- [x] Task 5: complete (commits 027fb39..ab1e090, fix round 1/5: 2 addressed, 0 open; review clean)
- Task 5: minor (deferred): unhashable enum value / non-serializable ticket_no can throw; allowlist/enum test coverage narrow; no MISS_TEXT/ERROR_TEXT classifier test; found:false applied beyond MCP; ticket JSON spacing; formatting imports faq constants
- [x] Task 6: complete (commits ab1e090..e1e9ec5, review clean; Minor notes parked: duration excludes formatting, first backoff 0s, custom hooks unguarded, unknown-tool audit DRY, `specs is not None`, deny duration ~0, test gaps for transient-retry-success/success-audit/MCP-read-policy)
- Resumed by user at 2026-11-19.
- Task 7: Ruling: the plan's `tool.args` schema assumption and `classify=None/format_result=None` are wrong for installed `langchain-core 1.8.1` / adapters 0.3.2; the spec's validation/formatting requirements are binding. Fix uses the full `inputSchema` and normalizes MCP content blocks. Cost if wrong: MCP tool metadata/result normalization changes.
- [x] Task 7: complete (commits e1e9ec5..734d299, fix round 1/5: 2 addressed, 0 open; review clean)
- Task 7: minor (deferred): conversion errors abort all servers; outer timeout discards partial results; MCP contextvar not propagated; delay test doesn't prove call-time env read; duplicate server-name lists; no end-to-end isError test
- Task 9 carry-over: prompt still says `不得调用 create_ticket`; Task 9 must replace it with the explicit-request/collect-info/preview/confirm rules.
- [x] Task 8: complete (commits 734d299..54dcb05, review clean; Minor notes parked: guard normalization discarded before engine, control fallback when spec missing, refund route ignores exposed flag, unbound registered tools executable, guard-audit/agent-level write-denial test gaps, ambient settings in binding tests, duplicate fixture, stale registry docstring)
- [x] Task 9: complete (commits 54dcb05..632de35, fix round 1/5: 1 addressed, 0 open; review clean)
- Task 9: minor (deferred): ch06 confirm glue untested; identity-based first-ticket matching; serializability test overstates; resume audit text keyed by type not pending kind; prompt 退款 tension
- Task 9: reviewer minor "failure string captured as ticket number" ruled non-issue: `classify_result` returns `error` for `创建工单失败…`, so `created_ticket_no` is not set.
- [x] Task 10: complete (commits 632de35..3473754, review clean; Minor notes parked: ch05 retries=0 not pinned/tested, duplicate 502 helper, missing-number tests only data=None, raw data returned, no SSE/resume fallback test)
- [x] Task 11: complete (commits 3473754..6766a85, fix round 1/5: 3 addressed, 0 open; review clean)
- Task 11: minor (deferred): specs=None path reopens allowlist hole for alternate callers; FAQ miss sentence drift; per-request MCP discovery on legacy; stale docstrings; redundant specs branch
- [x] Task 12: complete (commits 6766a85..bdca6a9, review clean; eval 64 cases, 63 correct, accuracy 0.984, 人工 2/2; Minor notes parked: dataset size assertion >=56, thin 人工 sample, stale 八类 docstring, report title says ch06)
- [x] Task 13: complete (commit bdca6a9..3165011, Vibe Coding exception: no review; JS syntax checked; preview card + confirm/cancel wired)
- Final review (opus): 4 Important — registry race, unbound-tool execution, query_promotions prompt/acceptance, Task 14 evidence; no Critical.
- Final fix wave: commit f673f40; 3 Important + minors addressed; full suite 772 passed; scoped re-review clean (2 residual Minor parked: malformed nested enum can throw; pattern error text only fits blank pattern).
- [x] Task 14: complete (acceptance report docs/superpowers/reports/2026-11-19-ch08-acceptance.md; delivery commit c5242d5; MySQL DDL apply and browser visual check left to user because local MySQL service access is denied)
- Final verdict: ready to merge with fixes applied; no open Critical/Important.
