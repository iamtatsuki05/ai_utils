from typing import Any, Optional

from google.generativeai import GenerativeModel
from tenacity import retry, stop_after_attempt, wait_exponential

from ai_utils.api_client.base import AsyncResource


class AsyncGoogleResource(AsyncResource):
    def __init__(
        self,
        client: GenerativeModel,
        generation_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.client = client
        self.generation_config = generation_config
        super().__init__(**kwargs)

    @retry(
        wait=wait_exponential(multiplier=60, min=60, max=240),
        stop=stop_after_attempt(3),
    )
    async def call(self, input_data: Any, **generation_config) -> str:
        return (
            await self.client.generate_content_async(
                messages=input_data,
                **generation_config,
            )
            .candidates[0]
            .content
        )

    async def __call__(self, input_data: Any, *args, **kwargs) -> str:
        return await self.task(input_data, *args, **kwargs)
