import os
from functools import lru_cache
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()


@lru_cache(maxsize=1)
def demo_chatbot():
    if PROVIDER == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Get a free key at https://console.groq.com "
                "and create a .env file with: GROQ_API_KEY=your_key_here"
            )
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.3,
            api_key=api_key,
            max_tokens=1024,
        )

    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            temperature=0.3,
            num_predict=1024,
        )

    raise ValueError("LLM_PROVIDER must be 'groq' or 'ollama'")


def approx_token_counter(messages):
    return sum(len(str(m.content)) // 4 + 4 for m in messages)


examples = [
    {
        "input": "Who are you?",
        "output": "I'm the BEPEC assistant. I help you with questions about our "
                  "AI, Data Science, and Data Engineering courses, placements, and careers.",
    },
    {
        "input": "I'm from a non-IT background. Can I learn AI?",
        "output": "Absolutely. Many BEPEC learners come from non-IT backgrounds. "
                  "We start from fundamentals and take you to job-ready, step by step. "
                  "No prior coding needed to begin.",
    },
    {
        "input": "How long does the course take?",
        "output": "It depends on the track and your pace, but most learners become "
                  "job-ready in a few focused months. Pick a track and commit to "
                  "consistent daily practice.",
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, friendly AI assistant built by BEPEC Solutions. "
            "You remember the ongoing conversation and answer clearly and concisely. "
            "If you don't know something, say so honestly instead of making things up. "
            "Match the tone and style of the examples. and end with Thanks for Asking BEPEC",
        ),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=approx_token_counter,
    include_system=False,
    start_on="human",
)


def demo_memory():
    return InMemoryChatMessageHistory()


def demo_conversation(input_text, memory):
    llm = demo_chatbot()

    base_chain = (
        RunnablePassthrough.assign(
            history=itemgetter("history") | trimmer
        )
        | CHAT_PROMPT
        | llm
    )

    chat_with_history = RunnableWithMessageHistory(
        base_chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="history",
    )

    try:
        result = chat_with_history.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": "bepec-session"}},
        )
        chat_reply = result.content
    except Exception as e:
        chat_reply = f"Sorry, something went wrong: {str(e)}"

    return chat_reply, memory