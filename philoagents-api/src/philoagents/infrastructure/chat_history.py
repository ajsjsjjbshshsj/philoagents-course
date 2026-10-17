
from datetime import datetime, timezone

from pymongo import MongoClient

from philoagents.config import settings


COLLECTION_NAME = "chat_history"


def save_turn(
        player_id: str,
        philosopher_id: str,
        user_message: str,
        assistant_message: str,
) -> None:
    """保存一轮完整的玩家与 NPC 对话。"""

    document = {
        "player_id": player_id,
        "philosopher_id": philosopher_id,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "created_at": datetime.now(timezone.utc),
    }

    with MongoClient(settings.MONGO_URI) as client:
        collection = client[settings.MONGO_DB_NAME][COLLECTION_NAME]
        collection.insert_one(document)


def get_turns(
        player_id: str,
        philosopher_id: str,
        limit: int = 100,
) -> list[dict]:
    """查询某位玩家与某位 NPC 最近的聊天记录。"""

    with MongoClient(settings.MONGO_URI) as client:
        collection = client[settings.MONGO_DB_NAME][COLLECTION_NAME]

        documents = list(
            collection.find(
                {
                    "player_id": player_id,
                    "philosopher_id": philosopher_id,
                },
                {
                    "_id": 0,
                    "user_message": 1,
                    "assistant_message": 1,
                    "created_at": 1,
                },
            )
            .sort("_id", -1)
            .limit(limit)
        )

    documents.reverse()

    for document in documents:
        document["created_at"] = (
            document["created_at"].isoformat()
        )

    return documents
