"""End-to-end smoke check against a running API.

Exercises the paths the game uses (REST, WebSocket streaming, RAG, two-turn memory)
and prints what the model actually said, so a human can judge the answer.

Run it with the services up (PYTHONPATH makes the `philoagents` package importable from
inside the api image, where it lives at /app/philoagents):

    docker compose exec -e PYTHONPATH=/app api /app/.venv/bin/python scripts/smoke_chat.py
"""

import asyncio
import json
import os
import re
import uuid

import httpx
import websockets
from philoagents.config import settings

BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
WS_URL = os.getenv("WS_URL", "ws://localhost:8000/ws/chat")
TIMEOUT = float(os.getenv("SMOKE_TIMEOUT", "180"))


def new_player_id() -> str:
    return f"smoke-{uuid.uuid4().hex[:12]}"


def rest_chat(player_id: str, philosopher_id: str, message: str) -> httpx.Response:
    response = httpx.post(
        f"{BASE}/chat",
        json={
            "message": message,
            "philosopher_id": philosopher_id,
            "player_id": player_id,
        },
        timeout=TIMEOUT,
    )
    print(f"  HTTP {response.status_code} {response.text[:500]}")
    return response


async def ws_chat(player_id: str, philosopher_id: str, message: str) -> str:
    chunks: list[str] = []
    async with websockets.connect(WS_URL, open_timeout=TIMEOUT) as ws:
        await ws.send(
            json.dumps(
                {"message": message, "philosopher_id": philosopher_id, "player_id": player_id}
            )
        )
        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=TIMEOUT))
            if "error" in event:
                raise AssertionError(f"websocket returned an error: {event['error']}")
            if "chunk" in event:
                chunks.append(event["chunk"])
            if event.get("streaming") is False and event.get("response") is not None:
                break
    print(f"  {len(chunks)} chunks -> {event['response'][:500]}")
    assert len(chunks) > 1, "expected the reply to arrive as several chunks"
    return event["response"]


def main() -> None:
    print(f"smoke target: {BASE}")

    print("\n[1] unknown philosopher is 404, not 500")
    probe = rest_chat(new_player_id(), "miguel", "hi")
    assert probe.status_code == 404, probe.text

    print("\n[2] oversized message is rejected before it reaches the model")
    big = rest_chat(new_player_id(), "socrates", "啊" * 2001)
    assert big.status_code == 422, big.text

    print("\n[3] a burst of requests trips the rate limiter (no LLM cost: unknown NPC)")
    spammer = new_player_id()
    codes = set()
    for _ in range(settings.RATE_LIMIT_MAX_MESSAGES + 1):
        codes.add(
            httpx.post(
                f"{BASE}/chat",
                json={"message": "hi", "philosopher_id": "miguel", "player_id": spammer},
                timeout=TIMEOUT,
            ).status_code
        )
    print(f"  status codes seen: {sorted(codes)}")
    assert codes == {404, 429}, codes

    player_id = new_player_id()
    print(f"\n[4] first turn, player_id={player_id}")
    first = rest_chat(player_id, "socrates", "你好，我叫小明，我正在研究塞尔的中文屋论证。")
    assert first.status_code == 200, first.text
    assert first.json()["response"].strip(), "empty reply"

    print("\n[5] second turn on the same thread: quoting the topic proves the player's")
    print("    message reached the model and the checkpoint remembered it")
    second = rest_chat(player_id, "socrates", "我刚才说我正在研究什么？")
    assert second.status_code == 200, second.text

    other_player = new_player_id()
    print(f"\n[6] a second player talks to the same NPC, for the reset check")
    rest_chat(other_player, "socrates", "你好，我是另一个人。")

    print("\n[7] retrieval-augmented turn for a philosopher with a corpus")
    rag = rest_chat(new_player_id(), "searle", "中文屋论证到底是什么，请依据资料解释")
    assert rag.status_code == 200, rag.text

    print("\n[8] websocket streaming, the path the game actually uses")
    asyncio.run(ws_chat(new_player_id(), "turing", "图灵测试能证明机器有意识吗？"))

    print("\n[9] websocket rejects a malformed frame instead of dropping the socket")
    asyncio.run(ws_reject_bad_frame())

    print("\n[10] /reset-memory clears one player, not the whole database")
    asyncio.run(reset_is_scoped(player_id, other_player))

    print("\nall smoke checks passed. Rows written under player_id='smoke-*'.")


async def ws_reject_bad_frame() -> None:
    async with websockets.connect(WS_URL, open_timeout=TIMEOUT) as ws:
        await ws.send(json.dumps({"message": "x", "philosopher_id": "socrates"}))
        event = json.loads(await asyncio.wait_for(ws.recv(), timeout=TIMEOUT))
    assert "error" in event, event
    print(f"  rejected with: {event['error']}")


async def reset_is_scoped(player_id: str, other_player: str) -> None:
    """Read the checkpoint collections through Mongo directly to prove the blast radius."""
    from pymongo import MongoClient

    db = MongoClient(settings.MONGO_URI)[settings.MONGO_DB_NAME]
    before = _thread_counts(db, player_id, other_player)
    print(f"  before: {before}")

    response = httpx.post(
        f"{BASE}/reset-memory", json={"player_id": player_id}, timeout=TIMEOUT
    )
    assert response.status_code == 200, response.text
    print(f"  reset says: {response.json()}")

    after = _thread_counts(db, player_id, other_player)
    print(f"  after:  {after}")
    assert after[player_id] == 0, f"{player_id} was not cleared: {after}"
    assert after[other_player] == before[other_player], "reset wiped another player"


def _thread_counts(db, *player_ids: str) -> dict:
    return {
        player_id: db["philosopher_state_checkpoints"].count_documents(
            {"thread_id": {"$regex": f"^{re.escape(player_id)}:"}}
        )
        for player_id in player_ids
    }


if __name__ == "__main__":
    main()
