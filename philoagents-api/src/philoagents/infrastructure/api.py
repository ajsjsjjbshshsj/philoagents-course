from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect,Query
from fastapi.middleware.cors import CORSMiddleware
from opik.integrations.langchain import OpikTracer
from pydantic import BaseModel,Field
from .chat_history import save_turn, get_turns
from philoagents.application.conversation_service.generate_response import (
    get_response,
    get_streaming_response,
)
from philoagents.application.conversation_service.reset_conversation import (
    reset_conversation_state,
)
from philoagents.domain.philosopher_factory import PhilosopherFactory

from .opik_utils import configure

configure()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the API."""
    # Startup code (if any) goes here
    yield
    # Shutdown code goes here
    opik_tracer = OpikTracer()
    opik_tracer.flush()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    message: str
    philosopher_id: str
    player_id: str=Field(min_length=8,max_length=100)
    philosopher_name: str=Field(min_length=8,max_length=100)



@app.post("/chat")
async def chat(chat_message: ChatMessage):
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
        opik_tracer = OpikTracer()
        opik_tracer.flush()

        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()


            if (
                    not isinstance(data, dict)
                    or "message" not in data
                    or "philosopher_id" not in data
                    or not isinstance(data.get("player_id"), str)
                    or not 8 <= len(data["player_id"]) <= 100
            ):
                await websocket.send_json({
                    "error": "Missing or invalid message, philosopher_id or player_id"
                })
                continue

            try:
                philosopher_factory = PhilosopherFactory()
                philosopher = philosopher_factory.get_philosopher(
                    data["philosopher_id"]
                )

                # Use streaming response instead of get_response
                response_stream = get_streaming_response(
                    messages=data["message"],
                    philosopher_id=data["philosopher_id"],
                    philosopher_name=philosopher.name,
                    philosopher_perspective=philosopher.perspective,
                    philosopher_style=philosopher.style,
                    philosopher_context="",
                    player_id=data["player_id"]
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
                        data["player_id"],
                        data["philosopher_id"],
                        data["message"],
                        full_response,
                    )

                await websocket.send_json(
                    {"response": full_response, "streaming": False}
                )

            except Exception as e:
                opik_tracer = OpikTracer()
                opik_tracer.flush()

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
        PhilosopherFactory.get_philosopher(philosopher_id)

        turns = await asyncio.to_thread(
            get_turns,
            player_id,
            philosopher_id,
        )

        return {"turns": turns}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/reset-memory")
async def reset_conversation():
    """Resets the conversation state. It deletes the two collections needed for keeping LangGraph state in MongoDB.

    Raises:
        HTTPException: If there is an error resetting the conversation state.
    Returns:
        dict: A dictionary containing the result of the reset operation.
    """
    try:
        result = await reset_conversation_state()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
