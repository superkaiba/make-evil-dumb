# Local patches

## `mcp-ssh-manager+3.2.2.patch`

Patches `~/.local/node_modules/mcp-ssh-manager/src/index.js` (npm package
`mcp-ssh-manager@3.2.2`) so the SSH MCP server picks up newly-provisioned
pods without an MCP process restart.

### Why

The upstream `loadServerConfig()` returns a snapshot of `servers` populated
once at process startup. Combined with the upstream parse regex
`/^SSH_SERVER_([A-Z0-9_]+)_HOST$/` — which rejects lowercase letters and
hyphens — this meant:

1. **Hot-reload bug.** `pod.py provision` writes the new pod into
   `~/.claude/mcp.json`, but the running MCP process keeps the old env in
   memory. New pods were invisible until the user manually `/mcp`-restarted.
2. **Regex bug.** Even after restart, ephemeral pods named like
   `epm-issue-261` (lowercase + hyphens) failed the upstream regex. Only
   `pod1..pod6` ever worked.

### What the patch changes

Replaces `loadServerConfig()` with a version that:

- Re-reads `~/.claude/mcp.json` on mtime change (path overridable via
  `SSH_HOTRELOAD_PATH` env var).
- Parses `mcpServers.ssh.env` with a permissive regex
  `/^SSH_SERVER_(.+)_HOST$/` that accepts any suffix.
- Falls back to the startup `servers` snapshot on any read/parse error
  (so a malformed mcp.json doesn't brick the MCP).

### How to apply

The patch is applied in-place on the node_modules install. To re-apply
after `npm install` (which would overwrite our edit), run:

```bash
patch -p1 -d ~/.local < patches/mcp-ssh-manager+3.2.2.patch
```

Or, for persistent application across reinstalls, install
[`patch-package`](https://github.com/ds300/patch-package) in
`~/.local/package.json` and copy this file to
`~/.local/patches/mcp-ssh-manager+3.2.2.patch`.

### Companion change in `pod_config.py`

`pod_config.py:_generate_mcp_env()` now writes env keys as
`SSH_SERVER_<UPPER_NAME>_HOST` (verbatim) instead of always prepending
`POD`. This produces clean keys like `SSH_SERVER_EPM-ISSUE-261_HOST` which
round-trip to the pod name `epm-issue-261` after lowercase. The strip
regex handles all three shapes (permanent, new ephemeral, legacy
ephemeral) so a one-time `--sync` cleans up.
