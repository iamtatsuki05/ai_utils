import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional


class AsyncResource(ABC):
    def __init__(self, concurrency: Optional[int] = 1) -> None:
        self.semaphore = asyncio.Semaphore(concurrency)

    async def task(self, *args, **kwargs) -> Any:
        async with self.semaphore:
            return await self.call(*args, **kwargs)

    @abstractmethod
    async def call(self, *args, **kwargs) -> Any:
        pass
