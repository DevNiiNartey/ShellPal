import os
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from openai import AsyncOpenAI

from client.responses import EventType, StreamEvent, TextDelta, TokenUsage

_ = load_dotenv()


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
    ) -> AsyncGenerator:
        client = self.get_client()
        kwargs = {
            "model": "stepfun/step-3.5-flash:free",
            "messages": messages,
            "stream": stream,
        }
        if stream:
            event = await self._stream_response(client, kwargs)
            yield event
        else:
            event = await self._non_stream_response(client, kwargs)
            yield event

    async def _stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]):
        pass

    async def _non_stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> StreamEvent:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        text_delta = None

        if message.content:
            text_delta = TextDelta(content=message.content)

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )
        return StreamEvent(
            type=EventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )
