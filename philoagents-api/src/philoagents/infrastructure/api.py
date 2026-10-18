import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from opik.integrations.langchain import OpikTracer
from pydantic import BaseModel, Field, ValidationError

from .chat_history import ensure_indexes, get_turns, save_turn
from .opik_utils import configure, tracing_enabled
from .rate_limit import allow
from philoagents.application.conversation_service.generate_response import (
    get_response,
    get_streaming_response,
    shutdown_workflow_graph,
)
from philoagents.application.conversation_service.reset_conversation import (
    reset_conversation_state,
)
from philoagents.config import settings
from philoagents.domain.exceptions import (
    PhilosopherContextNotFound,
    PhilosopherNameNotFound,
    PhilosopherPerspectiveNotFound,
    PhilosopherStyleNotFound,
)
from philoagents.domain.philosopher_factory import PhilosopherFactory

configure()


def flush_tracing() -> None:
    if tracing_enabled():
        OpikTracer().flush()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the API."""
    await asyncio.to_thread(ensure_indexes)
    yield
    await shutdown_workflow_graph()
    flush_tracing()


app = FastAPI(lifespan=lifespan)

_origins = settings.cors_origins
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        # A wildcard plus credentials is rejected by browsers, so never send both.
        allow_credentials="*" not in _origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class ChatMessage(BaseModel):
    message: str = Field(min_length=1, max_length=settings.MAX_MESSAGE_CHARS)
    philosopher_id: str = Field(min_length=1, max_length=64)
    player_id: str = Field(min_length=8, max_length=100)


class ResetRequest(BaseModel):
    player_id: str = Field(min_length=8, max_length=100)


def to_http_error(exc: Exception) -> HTTPException:
    if isinstance(
        exc,
        (
            PhilosopherNameNotFound,
            PhilosopherPerspectiveNotFound,
            PhilosopherStyleNotFound,
            PhilosopherContextNotFound,
        ),
    ):
        return HTTPException(status_code=404, detail=str(exc))

    return HTTPException(status_code=500, detail=str(exc))


@app.post("/chat")
async def chat(chat_message: ChatMessage):
    allowed, retry_after = allow(f"chat:{chat_message.player_id}")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many messages. Wait before sending another one.",
            headers={"Retry-After": str(retry_after)},
        )

    try:
        philosopher_factory = PhilosopherFactory()
        philosopher = philosopher_factory.get_philosopher(chat_message.philosopher_id)

        response, _ = await get_response(
            messages=chat_message.message,
            philosopher_id=chat_message.philosopher_id,
            philosopher_name=philosopher.name,
            philosopher_perspective=philosopher.perspective,
            philosopher_style=philosopher.style,
            philosopher_context="",
            player_id=chat_message.player_id
        )
        await asyncio.to_thread(
            save_turn,
            chat_message.player_id,
            chat_message.philosopher_id,
            chat_message.message,
            response,
        )
        return {"response": response}
    except Exception as e:
        raise to_http_error(e)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            try:
                payload = ChatMessage.model_validate(data)
            except ValidationError:
                await websocket.send_json({
                    "error": (
                        "message, philosopher_id and a 8-100 character player_id are required"
                    )
                })
                continue

            allowed, retry_after = allow(f"chat:{payload.player_id}")
            if not allowed:
                await websocket.send_json({
                    "error": f"Too many messages. Try again in {retry_after}s."
                })
                continue

            try:
                philosopher_factory = PhilosopherFactory()
                philosopher = philosopher_factory.get_philosopher(
                    payload.philosopher_id
                )

                # Use streaming response instead of get_response
                response_stream = get_streaming_response(
                    messages=payload.message,
                    philosopher_id=payload.philosopher_id,
                    philosopher_name=philosopher.name,
                    philosopher_perspective=philosopher.perspective,
                    philosopher_style=philosopher.style,
                    philosopher_context="",
                    player_id=payload.player_id
                )

                # Send initial message to indicate streaming has started
                await websocket.send_json({"streaming": True})

                # Stream each chunk of the response
                full_response = ""
                async for chunk in response_stream:
                    full_response += chunk
                    await websocket.send_json({"chunk": chunk})

                # AI 完整回复生成后，保存到 MongoDB
                if full_response.strip():
                    await asyncio.to_thread(
                        save_turn,
                        payload.player_id,
                        payload.philosopher_id,
                        payload.message,
                        full_response,
                    )

                await websocket.send_json(
                    {"response": full_response, "streaming": False}
                )

            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        pass


@app.get("/chat-history/{philosopher_id}")
async def read_chat_history(
        philosopher_id: str,
        player_id: str = Query(..., min_length=8, max_length=100),
):
    """获取当前玩家与指定哲学家的历史对话。"""

    try:
        # 验证 NPC 是否存在
        PhilosopherFactory().get_philosopher(philosopher_id)

        turns = await asyncio.to_thread(
            get_turns,
            player_id,
            philosopher_id,
        )

        return {"turns": turns}

    except Exception as e:
        raise to_http_error(e)


@app.post("/reset-memory")
async def reset_conversation(payload: ResetRequest):
    """Clears one player's conversation state.

    It used to drop the checkpoint collections outright, which reset every player in the
    database at once. Only the caller's own threads are removed now, so no secret is
    needed: the worst a caller can delete is the history of the player_id they present.

    Returns:
        dict: the player and how many documents each collection lost.
    """
    try:
        return await asyncio.to_thread(reset_conversation_state, payload.player_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
