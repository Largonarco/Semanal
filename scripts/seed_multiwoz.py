"""Seed the ingest API with MultiWOZ dialogues + synthetic timestamps.

Each MultiWOZ dialogue becomes one `POST /conversations` call (no conversation_id,
so each is a fresh conversation). Conversation start times are distributed
uniformly over the trailing N days; inter-turn gaps are uniform [30s, 5min].

Usage:
    docker compose exec api python -m scripts.seed_multiwoz --limit 500
"""
from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

log = logging.getLogger("seed_multiwoz")


def _reconstruct_turns(example: dict[str, Any]) -> list[dict[str, str]]:
    turns_field = example.get("turns") or {}
    speakers = turns_field.get("speaker") or []
    utterances = turns_field.get("utterance") or []
    out: list[dict[str, str]] = []
    for i, utt in enumerate(utterances):
        if not utt:
            continue
        speaker_code = speakers[i] if i < len(speakers) else 0
        role = "user" if speaker_code == 0 else "agent"
        out.append({"role": role, "content": utt})
    return out


def _attach_timestamps(
    turns: list[dict[str, str]],
    start: datetime,
    gap_min_s: float = 30.0,
    gap_max_s: float = 300.0,
) -> list[dict[str, Any]]:
    cursor = start
    out: list[dict[str, Any]] = []
    for turn in turns:
        out.append({**turn, "timestamp": cursor.isoformat()})
        cursor = cursor + timedelta(seconds=random.uniform(gap_min_s, gap_max_s))
    return out


def _post_conversation(
    client: httpx.Client, turns: list[dict[str, Any]], metadata: dict[str, Any]
) -> bool:
    payload = {"turns": turns, "metadata": metadata}
    try:
        r = client.post("/conversations", json=payload)
        r.raise_for_status()
        return True
    except httpx.HTTPError as e:
        log.warning("POST /conversations failed: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--anchor-date",
        default=None,
        help="ISO datetime (UTC) for the newest possible conversation start (default: now).",
    )
    parser.add_argument("--spread-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        default="multi_woz_v22",
        help="HuggingFace dataset identifier (default: multi_woz_v22)",
    )
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s"
    )
    random.seed(args.seed)

    anchor = (
        datetime.fromisoformat(args.anchor_date)
        if args.anchor_date
        else datetime.now(timezone.utc)
    )
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    earliest = anchor - timedelta(days=args.spread_days)

    log.info("Loading dataset %s split=%s ...", args.dataset, args.split)
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    n = min(args.limit, len(ds))
    log.info("Seeding %d dialogues to %s", n, args.api_url)

    successes = 0
    failures = 0
    with httpx.Client(base_url=args.api_url, timeout=60.0) as client:
        for i in range(n):
            example = ds[i]
            turns = _reconstruct_turns(example)
            if not turns:
                continue

            spread = (anchor - earliest).total_seconds()
            start = earliest + timedelta(seconds=random.random() * spread)
            turns_ts = _attach_timestamps(turns, start)

            metadata = {
                "source": args.dataset,
                "dialogue_id": example.get("dialogue_id"),
                "services": example.get("services", []),
            }
            if _post_conversation(client, turns_ts, metadata):
                successes += 1
            else:
                failures += 1

            if (i + 1) % 50 == 0:
                log.info(
                    "Progress %d/%d  success=%d  failures=%d",
                    i + 1,
                    n,
                    successes,
                    failures,
                )

    log.info("Done. success=%d failures=%d", successes, failures)


if __name__ == "__main__":
    main()
