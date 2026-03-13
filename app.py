"""Application entry point for testing the LLM client."""

import asyncio

from client.llm_client import LLMClient


class LLMENTRY:
    """Entry point class for LLM client demonstration."""

    @staticmethod
    async def entry():
        """Run a test chat completion and print the streaming events.

        Creates an LLM client, sends a test message, and prints each event
        received from the streaming response.
        """
        # Initialize the LLM client
        spin_up = LLMClient()

        # Define the conversation messages
        input = [
            {
                "role": "user",
                "content": "Why is the sky blue?",
            }
        ]

        # Stream the chat completion and print each event
        async for event in spin_up.chat_completion(input, True):
            print(event)


# Run the async entry point
asyncio.run(LLMENTRY.entry())
