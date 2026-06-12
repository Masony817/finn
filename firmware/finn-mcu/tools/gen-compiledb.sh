#!/usr/bin/env bash
#
# Generate one merged clangd compilation database covering every PlatformIO
# environment. clangd keys off per-file entries, so each script still sees only
# the headers its own env compiles against -- no shared/umbrella include path.
#
# Re-run this after adding an env, adding a source file, or changing lib_deps.
# Requires: pio, jq.

set -euo pipefail

cd "$(dirname "$0")/.."   # firmware/finn-mcu

# Discover all [env:NAME] sections from platformio.ini.
mapfile -t ENVS < <(grep -oE '^\[env:[^]]+\]' platformio.ini | sed -E 's/^\[env:(.+)\]/\1/')

if [ "${#ENVS[@]}" -eq 0 ]; then
  echo "No [env:*] sections found in platformio.ini" >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

for env in "${ENVS[@]}"; do
  echo "==> compiledb: $env"
  pio run -e "$env" -t compiledb >/dev/null
  cp compile_commands.json "$TMPDIR/cc_${env}.json"
done

# Merge all envs, dedup by file (keeps one entry per translation unit).
jq -s 'add | unique_by(.file)' "$TMPDIR"/cc_*.json > compile_commands.json

echo "==> wrote compile_commands.json ($(jq 'length' compile_commands.json) entries across ${#ENVS[@]} envs)"
