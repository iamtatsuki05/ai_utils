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


class AsyncOpenAISText2SpeechResource(AsyncResource):
    def __init__(
        self,
        client: Union[AsyncOpenAI, AsyncAzureOpenAI],
        model_name: Optional[str] = 'tts-1',
        voice: Optional[str] = 'alloy',
        generation_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.voice = voice
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
        voice: Optional[str] = None,
        **generation_config,
    ) -> str:
        return await self.client.audio.speech.create(
            model=self.model_name if model_name is None else model_name,
            voice=self.voice if voice is None else voice,
            file=input_data,
            **generation_config,
        ).text

    async def __call__(self, input_data: Any, *args, **kwargs) -> str:
        return await self.task(input_data, *args, **kwargs)
