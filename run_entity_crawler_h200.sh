#!/usr/bin/env bash
# ==============================================================================
# Ghana Search: Persistent Named Entity Cache Crawler Daemon
# Runs in the background on the H200 GPU instance to build an exhaustive offline
# cache across all 253,000+ Ghanaian named entities and auto-push to Hugging Face.
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

LOG_FILE="$PROJECT_DIR/data/crawler.log"
PID_FILE="$PROJECT_DIR/data/crawler.pid"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

mkdir -p "$PROJECT_DIR/data"

echo "=== Starting Ghana Search Named Entity Crawler ==="
echo "Project Directory: $PROJECT_DIR"
echo "Log File:          $LOG_FILE"
echo "PID File:          $PID_FILE"
echo "Python Executable: $VENV_PYTHON"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" || true)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[!] Crawler is already running with PID $OLD_PID"
        exit 0
    fi
fi

# Run the persistent auto-restarting crawler in background
nohup bash -c '
    while true; do
        echo "[$(date -u +"%Y-%m-%d %H:%M:%S UTC")] Launching Entity Cache Crawler..." >> "'"$LOG_FILE"'"
        "'"$VENV_PYTHON"'" -m ghana_mcp.entity_cache_crawler \
            --batch-size 500 \
            --max-wait 5.0 \
            --workers 3 >> "'"$LOG_FILE"'" 2>&1 || true
        echo "[$(date -u +"%Y-%m-%d %H:%M:%S UTC")] Crawler process exited. Restarting in 15s..." >> "'"$LOG_FILE"'"
        sleep 15
    done
' > /dev/null 2>&1 &

CRAWLER_PID=$!
echo "$CRAWLER_PID" > "$PID_FILE"

echo "[+] Daemon started in background with PID $CRAWLER_PID"
echo "[+] To monitor live progress: tail -f data/crawler.log"
echo "[+] To check status: $VENV_PYTHON -m ghana_mcp.entity_cache_crawler --status"
