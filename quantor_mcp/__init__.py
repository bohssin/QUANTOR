"""QUANTOR engine MCP server. Plan §14.2.

Named `quantor_mcp` rather than `mcp` deliberately: the MCP SDK owns the `mcp`
top-level name, and a local package shadowing it breaks the server's own import.
"""
