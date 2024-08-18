"""llama.cpp server backend support"""

from unreasonable_llama import UnreasonableLlama


class LlamaBackend:
    def __init__(self, server_url: str, request_timeout: int) -> None:
        self.llama = UnreasonableLlama(server_url, request_timeout)
