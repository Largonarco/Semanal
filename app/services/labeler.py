from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk, Topic
from app.db.session import session_scope

log = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an analyst classifying clusters of conversation snippets for a "
    "product manager.\n"
    "Given representative snippets from a single cluster, produce a JSON object with:\n"
    '- "topic": concise free-text label naming the underlying topic (3-8 words)\n'
    '- "sentiment": one of "positive", "negative", "neutral", "mixed" — '
    "based on the speakers' attitudes toward the subject of discussion\n"
    '- "pm_insight": a 1-2 sentence actionable insight a PM could use, in PM voice\n'
    '- "supporting_snippets": 3-5 short verbatim quotes from the input that illustrate the topic\n'
    "\n"
    "Return ONLY the JSON object, no other text."
)

VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}


def _build_user_prompt(snippets: list[str]) -> str:
    bulleted = "\n".join(f"- {s}" for s in snippets)
    return (
        "Here are representative snippets from one cluster:\n\n"
        f"{bulleted}\n\n"
        "Produce the analysis JSON."
    )


def _make_bedrock_client():
    settings = get_settings()
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
    if settings.aws_secret_access_key:
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("bedrock-runtime", **kwargs)


def _mock_label(snippets: list[str]) -> dict[str, Any]:
    sample = (snippets[0] if snippets else "")[:80]
    return {
        "topic": f"mock-topic-{sample[:32]}",
        "sentiment": "neutral",
        "pm_insight": "Mocked label — set MOCK_LLM=false and provide AWS creds to call Bedrock.",
        "supporting_snippets": snippets[:3],
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first balanced { ... } object out of the model output.

    Sonnet 4.6 on Bedrock doesn't accept assistant-message prefill, so we rely
    on the system prompt instructing JSON-only output and tolerate leading
    whitespace / minor prose around the object.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Bedrock response contained no JSON object: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _invoke_bedrock_sync(snippets: list[str]) -> dict[str, Any]:
    settings = get_settings()
    client = _make_bedrock_client()
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": _build_user_prompt(snippets)},
        ],
    }
    response = client.invoke_model(
        modelId=settings.bedrock_model_id,
        body=json.dumps(body),
    )
    payload = json.loads(response["body"].read())
    text = payload["content"][0]["text"]
    return _extract_json_object(text)


async def _fetch_topic_and_snippets(
    session: AsyncSession, topic_id: uuid.UUID, n: int = 20
) -> tuple[Topic | None, list[str]]:
    topic = await session.get(Topic, topic_id)
    if topic is None or topic.centroid is None:
        return None, []
    stmt = (
        select(Chunk.content)
        .where(Chunk.topic_id == topic_id)
        .order_by(Chunk.embedding.cosine_distance(topic.centroid))
        .limit(n)
    )
    snippets = [r.content for r in (await session.execute(stmt)).all()]
    return topic, snippets


def _coerce_label(parsed: dict[str, Any]) -> dict[str, Any]:
    label = (parsed.get("topic") or "(unlabeled)").strip()
    sentiment_raw = (parsed.get("sentiment") or "neutral").strip().lower()
    sentiment = sentiment_raw if sentiment_raw in VALID_SENTIMENTS else "mixed"
    pm_insight = (parsed.get("pm_insight") or "").strip()
    supporting = parsed.get("supporting_snippets") or []
    if not isinstance(supporting, list):
        supporting = []
    return {
        "topic": label[:500],
        "sentiment": sentiment,
        "pm_insight": pm_insight,
        "supporting_snippets": [str(s)[:1000] for s in supporting[:5]],
    }


async def label_topic_via_bedrock(topic_id: uuid.UUID) -> dict[str, Any]:
    settings = get_settings()

    async with session_scope() as session:
        topic, snippets = await _fetch_topic_and_snippets(session, topic_id)
        if topic is None:
            return {"status": "topic_missing", "topic_id": str(topic_id)}
        if not snippets:
            return {"status": "no_snippets", "topic_id": str(topic_id)}

    use_mock = settings.mock_llm or not settings.has_aws_credentials
    if use_mock:
        parsed = _mock_label(snippets)
    else:
        try:
            parsed = await asyncio.to_thread(_invoke_bedrock_sync, snippets)
        except Exception:
            log.exception("Bedrock invocation failed for topic %s", topic_id)
            raise

    coerced = _coerce_label(parsed)

    async with session_scope() as session:
        topic = await session.get(Topic, topic_id)
        if topic is None:
            return {"status": "topic_missing_post_label", "topic_id": str(topic_id)}
        topic.label = coerced["topic"]
        topic.sentiment = coerced["sentiment"]
        topic.pm_insight = coerced["pm_insight"]
        topic.last_labeled_at = datetime.now(timezone.utc)

    return {
        "status": "labeled",
        "topic_id": str(topic_id),
        **coerced,
        "mocked": use_mock,
    }
