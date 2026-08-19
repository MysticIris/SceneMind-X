"""ASGI entry point for the Phase 1 server."""

from scenemindx.phase1.api import create_app

app = create_app()
