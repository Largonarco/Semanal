from __future__ import annotations

import os

# Pre-set env vars so importing app.config inside tests doesn't fail trying to
# resolve database/redis connectivity from a missing .env file.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("MOCK_LLM", "true")
