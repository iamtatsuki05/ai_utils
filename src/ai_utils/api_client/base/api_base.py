from abc import ABC, abstractmethod
from typing import Any, Optional


class APIBase(ABC):
    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, client: Any = None):
        pass

    @abstractmethod
    def post_request(self, input_data: Any) -> Any:
        pass

    @abstractmethod
    def get_response(self, request_id: str) -> Any:
        pass

    @abstractmethod
    def get_status(self, request_id: str) -> Any:
        pass
