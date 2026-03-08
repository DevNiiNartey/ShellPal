"""LLM client module for handling chat completions with OpenAI-compatible APIs."""

import os
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from openai import AsyncOpenAI

from client.responses import EventType, StreamEvent, TextDelta, TokenUsage

# Load environment variables from .env file
_ = load_dotenv()


class LLMClient:
    """Async client for LLM chat completions with streaming support.

    This client wraps the AsyncOpenAI client to provide a simplified interface
    for chat completions, supporting both streaming and non-streaming modes.
    It handles connection pooling and proper resource cleanup.
    """
    def __init__(self) -> None:
        """Initialize the LLM client with no active connection."""
        self._client: AsyncOpenAI | None = None

    def get_client(self) -> AsyncOpenAI:
        """Get or create the AsyncOpenAI client instance.

        Lazily initializes the client on first call using environment variables:
        - OPENAI_API_KEY: API key for authentication
        - OPENAI_BASE_URL: Optional base URL for API endpoint

        Returns:
            AsyncOpenAI: The configured async OpenAI client.
        """
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        return self._client

    async def close(self) -> None:
        """Close the client connection and release resources."""
        if self._client:
            await self._client.close()
            self._client = None

    async def chat_completion(
        self, messages: list[dict[str, Any]], stream: bool = True
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send a chat completion request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            stream: Whether to stream the response (default: True).

        Yields:
            StreamEvent: Events containing text deltas, finish reasons, and token usage.
        """
        client = self.get_client()
        kwargs = {
            "model": "stepfun/step-3.5-flash:free",
            "messages": messages,
            "stream": stream,
        }
        # Route to appropriate handler based on streaming mode
        if stream:
            async for event in self._stream_response(client, kwargs):
                yield event
        else:
            event = await self._non_stream_response(client, kwargs)
            yield event

    async def _stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Handle streaming chat completion responses.

        Args:
            client: The AsyncOpenAI client instance.
            kwargs: Arguments passed to the chat.completions.create() call.

        Yields:
            StreamEvent: Events for each chunk received from the stream.
        """
        # Initiate the streaming request
        response = await client.chat.completions.create(**kwargs)

        # Process each chunk from the streaming response
        async for chunk in response:
            usage: TokenUsage | None = None
            finish_reason: str | None = None

            # Extract token usage if available in this chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens,
                )

            # Skip chunks without choices (e.g., usage-only chunks)
            if not chunk.choices:
                continue

            # Extract content delta from the first choice
            choice = chunk.choices[0]
            delta = choice.delta

            # Capture finish reason when generation completes
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            # Wrap the text content in a TextDelta object
            text_delta = None
            if delta:
                text_delta = TextDelta(content=delta.content)

            # Yield the event with extracted data
            yield StreamEvent(
                type=EventType.MESSAGE_COMPLETE,
                text_delta=text_delta,
                finish_reason=finish_reason,
                usage=usage,
            )

    async def _non_stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> StreamEvent:
        """Handle non-streaming chat completion response.

        Args:
            client: The AsyncOpenAI client instance.
            kwargs: Arguments passed to the chat.completions.create() call.

        Returns:
            StreamEvent: A single event containing the complete response.
        """
        # Send the request and wait for the complete response
        response = await client.chat.completions.create(**kwargs)

        # Extract the generated message from the first choice
        choice = response.choices[0]
        message = choice.message
        text_delta = None

        # Wrap the message content in a TextDelta object
        if message.content:
            text_delta = TextDelta(content=message.content)

        # Extract token usage information
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )

        # Return a single event with the complete response
        return StreamEvent(
            type=EventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )
