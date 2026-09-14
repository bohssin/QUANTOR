# QUANTOR engine MCP server

The agent reaches the engine only through these tools (plan §14.2). It does not
write backtest scripts, does not parse stdout, and cannot change the fill model,
the cost assumptions or the fold geometry.

## Run

    pip install mcp numpy numba pyarrow pandas
    python3 -m quantor_mcp.server        # stdio

## Register with Claude Code

`.mcp.json` in the repo root:

```json
{
  "mcpServers": {
    "quantor-engine": {
      "command": "python3",
      "args": ["/ABSOLUTE/PATH/TO/QUANTOR/quantor_mcp/server.py"]
    },
    "luxalgo": { "type": "http", "url": "https://mcp.luxalgo.com/mcp" }
  }
}
```

Or register the Library directly:

    claude mcp add --transport http luxalgo https://mcp.luxalgo.com/mcp

**Use the absolute script path, not `-m quantor_mcp.server` with a `cwd` field.**
The `cwd` was not applied in testing and the module is only importable from the
repo root, so the server died with `CONNECTION_CLOSED` (§19, Probe 9). The script
puts its own root on `sys.path`, so by-path works from anywhere. Startup is ~1 s.

Then scope the agent to the tools rather than the shell:

    claude -p "..." --mcp-config .mcp.json \
      --allowedTools "mcp__quantor-engine__*,mcp__luxalgo__*,Read,Edit"

Read the **`system/init`** event for `mcp_servers` — each entry carries a
`status`:

```json
"mcp_servers": [{"name": "quantor-engine", "status": "connected"},
                {"name": "luxalgo",        "status": "needs-auth"}]
```

A server that is configured but not `connected` is silent otherwise: the agent
is not told, it simply has fewer tools and gives a worse answer for a reason
nobody sees. Check the status; do not assume configured means available.

## Tools

| Tool | Returns |
|---|---|
| `data_list` | loaded sources: kind, resolution, span, tick size |
| `data_load` | loads a bar CSV, returns the quality report |
| `strategy_save` | new version; **runs the §9 validator first** |
| `strategy_list` / `strategy_get` | versions and their source |
| `backtest_run` | metrics, ambiguity rate, ledger reconciliation check |
| `optimize_run` | top results **and the comparison count** (§11) |
| `validate_run` | walk-forward per-fold results and efficiency |
| `chart_apply` | that run's own entries and exits, for the chart |

## Two layers, neither a sandbox

`strategy_save` runs `engine.signal.validate_signal_block` — an AST walk that
rejects imports, file access, `exec`/`eval`, dunder attribute access and forward
indexing. Execution then happens in a restricted namespace exposing only bars,
params and the indicator set.

**Neither is a security sandbox.** Python cannot be sandboxed by withholding
names, and numba's dispatchers need `__import__` at call time regardless — which
is exactly why the validator, not the namespace, is the enforcement. These are
correctness boundaries against an agent following the §9 contract imperfectly.
Never run a signal block the owner did not initiate.

## Errors are returned, never swallowed

Every tool catches its own exceptions and returns
`{"error": "KeyError: 'fast'", "hint": "..."}` as text.

This is not defensive style, it is a finding. The MCP SDK wraps any exception a
tool raises as `UnexpectedToolError: Error executing tool <name>` and the real
message never reaches the client. Tested against a real agent session, that made
every failure look identical — its own broken strategy, a dummy strategy, and a
nonexistent id all returned the same opaque string — so the agent correctly
concluded from the evidence that the tool was broken and stopped. It was not.

**An agent cannot fix a mistake it cannot see.** With error text returned, the
identical session went from 28 turns and giving up to 13 turns and a completed
backtest, self-correcting twice on the way. Keep the decorator on every tool.

## State

Data and strategies are in-process for now. The real build puts them in the
library (§16) so versions, lineage and runs survive a restart.
