from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain_openai import ChatOpenAI

from philoagents.application.conversation_service.workflow.tools import tools
from philoagents.config import settings
from philoagents.domain.prompts import (
    CONTEXT_SUMMARY_PROMPT,
    EXTEND_SUMMARY_PROMPT,
    PHILOSOPHER_CHARACTER_CARD,
    SUMMARY_PROMPT,
)



def get_chat_model(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.environ["QWEN_API_KEY"],
        base_url=os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ),
        model=os.getenv("QWEN_MODEL", "qwen3.8-flash"),
        temperature=temperature,
        extra_body={"enable_thinking": False},
    )



def get_philosopher_response_chain():
    model = get_chat_model()
    model = model.bind_tools(tools)
    system_message = PHILOSOPHER_CHARACTER_CARD

    prompt = ChatPromptTemplate.from_messages(
        [
            
    (
    "system",
    system_message.prompt
    + "\n\nReply in the language of the user's latest message. "
      "If the user writes Chinese, use Simplified Chinese. "
      "If the user writes English, use English. "
      "Remain in character and keep game dialogue concise."
    ),

        ],
        template_format="jinja2",
    )

    return prompt | model


def get_conversation_summary_chain(summary: str = ""):
    model = get_chat_model()

    summary_message = EXTEND_SUMMARY_PROMPT if summary else SUMMARY_PROMPT

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("human", summary_message.prompt),
        ],
        template_format="jinja2",
    )

    return prompt | model


def get_context_summary_chain():
    model = get_chat_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", CONTEXT_SUMMARY_PROMPT.prompt),
        ],
        template_format="jinja2",
    )

    return prompt | model