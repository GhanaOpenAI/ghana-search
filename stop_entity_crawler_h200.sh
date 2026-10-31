#!/usr/bin/env bash
# Stop script for Ghana Search Entity Cache Crawler Daemon
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$PROJECT_DIR/data/crawler.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "[*] Stopping crawler daemon (PID $PID)..."
        kill "$PID" || true
        # Also kill any child python crawler processes
        pkill -f "ghana_mcp.entity_cache_crawler" || true
        rm -f "$PID_FILE"
        echo "[+] Crawler daemon stopped."
        exit 0
    fi
fi

# Fallback cleanup
pkill -f "ghana_mcp.entity_cache_crawler" || true
rm -f "$PID_FILE"
echo "[*] No active crawler PID found (cleaned up orphan processes if any)."
