"""
PrivSynth Vercel Serverless Function Entry Point.

Exports the FastAPI ASGI 'app' instance so Vercel's Python runtime
and serverless builders can serve PrivSynth API endpoints.

This module adds the repo root to sys.path so the internal 'privsynth'
and 'backend' packages can be imported cleanly.
"""
import os
import sys

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import the configured FastAPI application from backend.main
from backend.main import app as _backend_app

# Explicit top-level assignment for Vercel AST entrypoint detection
app = _backend_app

# Vercel ASGI application export
__all__ = ["app"]
