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
    "quantor-engine": { "command": "python3", "args": ["-m", "quantor_mcp.server"] },
    "luxalgo":        { "command": "npx", "args": ["-y", "@luxalgo/mcp"] }
  }
}
```

Then scope the agent to the tools rather than the shell:

    claude -p "..." --mcp-config .mcp.json \
      --allowedTools "mcp__quantor-engine__*,mcp__luxalgo__*,Read,Edit"

Check `system/init` for `mcp_server_errors` — a server that failed to load is
silent otherwise.

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

## State

Data and strategies are in-process for now. The real build puts them in the
library (§16) so versions, lineage and runs survive a restart.
