import asyncio

from uuid import UUID
from langchain.schema import LLMResult
from typing import List, Any, Optional
from langchain.callbacks.base import AsyncCallbackHandler

## Langchain usa AsyncCallbackHandler para el streaming de tokens
class AsyncQueueCallbackHandler(AsyncCallbackHandler):
    """ callback handler where you can pass a queue and receive the tokens over the queue"""
    def __init__(self, queue: asyncio.Queue, cancelToken=None):
        self.__queue = queue
        self.__cancelToken = cancelToken
        self.__buffer = []

    def get_buffer_contents(self):
        return "".join(self.__buffer)

    def reset_buffer(self):
        self.__buffer = []

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self.__cancelToken and self.__cancelToken.isCancelled is True:
            raise asyncio.CancelledError

        if token is not None:
            self.__buffer.append(token)
            await self.__queue.put(token)

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None, **kwargs: Any,) -> None:
        await self.__queue.put("<END OF LLM RESPONSE>")
