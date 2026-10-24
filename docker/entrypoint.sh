#!/bin/sh
# desk   - API + UI on 0.0.0.0:8765 (set DESK_TOKEN; the path jail is on for non-loopback binds)
# worker - headless resumable hunt with checkpoints (see tool/worker.py)
# *      - any other command, e.g. "python run.py hunt --sector 15"
set -e
case "$1" in
  desk)
    shift
    if [ -n "$STATE_BUCKET" ] || [ -n "$STATE_GIT_BRANCH" ]; then
      python -m tool.checkpoint pull --data "$DESK_DATA" || true
    fi
    exec python -m tool --host 0.0.0.0 --port "${PORT:-8765}" --data "$DESK_DATA" "$@"
    ;;
  worker)
    shift
    exec python -m tool.worker --data "$DESK_DATA" "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
