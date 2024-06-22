from typing import Any, Optional, Union

import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ai_utils.api_client.base import AsyncResource


class AsyncOpenAIVisionResource(AsyncResource):
    def __init__(
        self,
        client: Union[AsyncOpenAI, AsyncAzureOpenAI],
        model_name: Optional[str] = 'dall-e-3',
        size: Optional[str] = '1024x1024',
        quality: Optional[str] = 'standard',
        n: Optional[int] = 1,
        generation_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.size = size
        self.quality = quality
        self.n = n
        self.generation_config = generation_config
        super().__init__(**kwargs)

    @retry(
        retry=retry_if_exception_type(
            (openai.error.APIConnectionError, openai.error.RateLimitError)
        ),
        wait=wait_exponential(multiplier=60, min=60, max=240),
        stop=stop_after_attempt(3),
    )
    async def call(
        self,
        input_data: Any,
        model_name: Optional[str] = None,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: Optional[int] = None,
        **generation_config,
    ) -> str:
        return (
            await self.client.images.generate(
                model=self.model_name if model_name is None else model_name,
                prompt=input_data,
                size=self.size if size is None else size,
                quality=self.quality if quality is None else quality,
                n=self.n if n is None else n,
                **generation_config,
            )
            .data[0]
            .url
        )

    async def __call__(self, input_data: Any, *args, **kwargs) -> str:
        return await self.task(input_data, *args, **kwargs)
