from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api import clustering_runs, conversations, health, jobs, topics
from app.api.errors import register_exception_handlers
from app.config import get_settings

_settings = get_settings()

logging.basicConfig(
    level=_settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sentiment Analytics Engine",
        description="Chunk-and-cluster sentiment analytics for conversational AI",
        version="0.1.0",
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(conversations.router)
    app.include_router(clustering_runs.router)
    app.include_router(topics.router)
    app.include_router(jobs.router)

    return app


app = create_app()
