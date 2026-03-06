import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from client.responses import EventType, StreamEvent, TextDelta

began = load_dotenv()


class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def chat_completion(
        self, messages: list[dict[str, Any]], stream: bool = True
    ):
        client = self.get_client()
        kwargs = {
            "model": "stepfun/step-3.5-flash:free",
            "messages": messages,
            "stream": stream,
        }
        if stream:
            await self._stream_response(client, kwargs)
        else:
            await self._non_stream_response(client, kwargs)

    async def _stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]):
        pass

    async def _non_stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> None:
        response: ChatCompletion = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        text_delta = None

        if message.content:
            text_delta = TextDelta(content=message.content)

        print(response)
