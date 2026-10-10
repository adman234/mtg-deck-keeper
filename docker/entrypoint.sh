#!/bin/sh
# Runs as root only long enough to fix data-folder ownership, then drops to PUID:PGID
# (Unraid's default is 99:100, i.e. nobody:users).
set -e

PUID="${PUID:-99}"
PGID="${PGID:-100}"
DATA_DIR="${DATA_DIR:-/app/data}"
set -- uvicorn deck_keeper.main:app --host 0.0.0.0 --port "${PORT:-9925}" \
  --proxy-headers --forwarded-allow-ips="*" --log-level "${LOG_LEVEL:-info}"

if [ "$(id -u)" = "0" ]; then
  mkdir -p "$DATA_DIR"
  if [ "$(stat -c %u:%g "$DATA_DIR")" != "$PUID:$PGID" ]; then
    echo "Setting owner of $DATA_DIR to $PUID:$PGID"
    chown -R "$PUID:$PGID" "$DATA_DIR"
  fi
  exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups "$@"
fi
exec "$@"
