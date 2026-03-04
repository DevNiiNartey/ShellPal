from ast import Return

from openai import AsyncOpenAI


class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key="", base_url="")
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
