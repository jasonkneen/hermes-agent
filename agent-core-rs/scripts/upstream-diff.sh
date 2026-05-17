#!/usr/bin/env bash
# upstream-diff.sh — show what's changed upstream in paths we actually track.
#
# Usage:
#   ./scripts/upstream-diff.sh                 # since the pinned UPSTREAM_REV
#   ./scripts/upstream-diff.sh <since-sha>     # since an arbitrary SHA
#   ./scripts/upstream-diff.sh --files-only    # just list changed files
#   ./scripts/upstream-diff.sh --bump <sha>    # update UPSTREAM_REV to <sha> (call after review)
#
# Reads tracked path globs from UPSTREAM_PATHS.md (the "Path mapping" table).
# Anything not in that list is filtered out as deliberately-stripped noise.

set -euo pipefail

cd "$(dirname "$0")/.."   # agent-core-rs/
ROOT="$(cd .. && pwd)"
PIN_FILE="UPSTREAM_REV"
PATHS_FILE="UPSTREAM_PATHS.md"

# Tracked paths (kept in sync with UPSTREAM_PATHS.md). Globs are
# resolved as pathspecs by `git log -- <pathspec>` so wildcards work.
TRACKED_PATHS=(
  # Provider adapters (wire format)
  "agent/anthropic_adapter.py"
  "agent/codex_responses_adapter.py"
  # Agent loop
  "run_agent.py"
  # Tool registry + built-ins we ship
  "tools/registry.py"
  "tools/__init__.py"
  "tools/terminal_tool.py"
  "tools/file_tools.py"
  "tools/file_operations.py"
  "tools/todo_tool.py"
  "tools/web_tools.py"
  # Gateway HTTP API
  "gateway/platforms/api_server.py"
  "gateway/session.py"
  # CLI flag surface
  "hermes_cli/main.py"
  "hermes_cli/oneshot.py"
  "hermes_cli/gateway.py"
  "hermes_cli/config.py"
)

mode="diff"
since=""
bump_to=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --files-only) mode="files"; shift ;;
    --bump)       bump_to="${2:?missing SHA}"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# //; s/^#//'
      exit 0
      ;;
    *) since="$1"; shift ;;
  esac
done

if [[ -n "$bump_to" ]]; then
  # Resolve to full SHA and update the pin.
  full=$(cd "$ROOT" && git rev-parse "$bump_to")
  echo "$full" > "$PIN_FILE"
  echo "UPSTREAM_REV → $full"
  exit 0
fi

if [[ -z "$since" ]]; then
  if [[ ! -f "$PIN_FILE" ]]; then
    echo "no $PIN_FILE; pass a SHA explicitly: $0 <since-sha>" >&2
    exit 2
  fi
  since="$(tr -d '[:space:]' < "$PIN_FILE")"
fi

cd "$ROOT"

# Validate the pinned SHA exists. If it doesn't, the user likely needs to
# `git fetch origin main` first.
if ! git cat-file -e "$since" 2>/dev/null; then
  echo "error: SHA $since not found locally. Try: git fetch origin main" >&2
  exit 2
fi

# How far behind are we?
head_sha="$(git rev-parse HEAD)"
ahead_behind="$(git rev-list --left-right --count "$since"...HEAD | awk '{print $1}')"
echo "# upstream-diff: $since..HEAD ($ahead_behind commits in window)"
echo "# tracked paths: ${#TRACKED_PATHS[@]} files/globs (see agent-core-rs/UPSTREAM_PATHS.md)"
echo

if [[ "$mode" == "files" ]]; then
  git log --name-only --pretty=format: "$since"..HEAD -- "${TRACKED_PATHS[@]}" \
    | grep -v '^$' \
    | sort -u
  exit 0
fi

# Full commit-by-commit summary with per-file diffs of tracked paths only.
git_log_out="$(git log --oneline "$since"..HEAD -- "${TRACKED_PATHS[@]}" || true)"
if [[ -z "$git_log_out" ]]; then
  echo "no tracked-path changes since pinned SHA. We're up to date."
  exit 0
fi

echo "## Commits touching tracked paths"
echo
echo "$git_log_out"
echo

echo "## Per-area summary"
echo
for path in "${TRACKED_PATHS[@]}"; do
  count="$(git log --oneline "$since"..HEAD -- "$path" | wc -l | tr -d ' ')"
  if [[ "$count" -gt 0 ]]; then
    printf '  %-46s %s commit(s)\n' "$path" "$count"
  fi
done
echo

echo "## Diffs (tracked paths only)"
echo
git log --reverse --pretty=format:'### %h %s%n' --patch "$since"..HEAD -- "${TRACKED_PATHS[@]}"

echo
echo "---"
echo "When you've reviewed and ported what matters, advance the pin:"
echo "  ./scripts/upstream-diff.sh --bump $head_sha"
