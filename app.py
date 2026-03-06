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

        await spin_up.chat_completion(input, False)


asyncio.run(LLMENTRY.entry())
