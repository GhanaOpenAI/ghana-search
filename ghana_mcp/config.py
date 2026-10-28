"""Configuration and constants for Ghana-MCP."""
import os
import socket

# Optional Gemini API key (defaults to empty since we use self-hosted EmbeddingGemma + BGE-Reranker)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SERVER_HOST = os.environ.get("HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("PORT", "8080"))

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
    "Ghana-MCP/1.0 (+https://github.com/GhanaOpenAI/ghana-mcp)"
)

# Strict 4.0s hard cap across all operations and sockets
MAX_SEARCH_SECONDS = 4.0
PROVIDER_TIMEOUT = 3.2  # maximum time to wait for external parallel providers
DEFAULT_TIMEOUT = 3.0   # seconds per individual external API socket query
socket.setdefaulttimeout(DEFAULT_TIMEOUT)

