from typing import Optional, Union

import openai
import tiktoken

from ai_utils.env import OPENAI_API_KEY


def setup_openai_client(
    api_key: Optional[str] = OPENAI_API_KEY,
    platform: Optional[str] = 'openai',
    azure_base_url: Optional[str] = None,
    api_version: Optional[str] = None,
) -> None:
    openai.api_key = api_key
    if platform == 'azure':
        openai.api_base = azure_base_url

        openai.api_version = api_version
        openai.api_type = 'azure'
        openai.api_key = api_key


def calc_tiktoken_length(
    text: str,
    model_id_or_encoder_name_or_tokenizer: Optional[
        Union[str, tiktoken.core.Encoding]
    ] = 'gpt-4',
) -> int:
    if isinstance(model_id_or_encoder_name_or_tokenizer, str):
        tokenizer = tiktoken.encoding_for_model(model_id_or_encoder_name_or_tokenizer)
    else:
        tokenizer = model_id_or_encoder_name_or_tokenizer

    return len(tokenizer.encode(text))
