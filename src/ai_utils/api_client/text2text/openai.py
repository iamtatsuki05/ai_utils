from typing import Any, Optional, AsyncIterator

from openai import AsyncOpenAI

from ai_utils.api_client.base.api_base import APIBase


class OpenAIAPIText2Text(APIBase):
    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
        model_name: Optional[str] = 'gpt-3.5-turbo',
    ) -> None:
        self.__api_key = api_key
        self.client = client if client is not None else AsyncOpenAI(api_key=self.__api_key)

    def post_request(self, input_data: Any) -> Any:
        return self.client.create_completion(**input_data)

    def get_response(self, request_id: str) -> Any:
        return self.client.get_completion(request_id)


    def _preprocess(self, input_data: Any) -> Any:
        return input_data

    def _postprocess(self, output_data: Any) -> Any:
        return output_data

    def call(self, input_data: Any) -> Any:
        self._preprocess(input_data)
        output_data = self.post_request(input_data)
        return self._postprocess(output_data)

    def __call__(self, input_data: Any) -> Any:
        return self.call(input_data)
