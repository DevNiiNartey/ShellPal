import asyncio

from client.llm_client import LLMClient


class LLMENTRY:
    @staticmethod
    async def entry():
        spin_up = LLMClient()
        input = [
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ]

        async for event in spin_up.chat_completion(input, False):
            print(event)


asyncio.run(LLMENTRY.entry())
