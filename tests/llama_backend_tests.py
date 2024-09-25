from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from unreasonable_llama import LlamaCompletionResponse

from unllamabot.llama_backend import LlamaBackend, LlamaCompletionRequest


@dataclass
class LlamaProps:
    model_name: str


def find_next_separator(string: str, separators: str) -> int | None:
    for i, char in enumerate(string):
        if char in separators:
            return i
    return None


class LlamaMock:
    def __init__(self, timeout: int) -> None:
        self.timeout = timeout
        self.mock_is_alive = True
        self.mock_response = ""

    async def get_streamed_completion(self, request: LlamaCompletionRequest) -> AsyncIterator[LlamaCompletionResponse]:
        words = self.mock_response
        separator_index = find_next_separator(words, " \n")
        while separator_index is not None:
            yield LlamaCompletionResponse(content=words[: separator_index + 1], id_slot=0, stop=False, index=0)
            words = words[separator_index + 1 :]
            separator_index = find_next_separator(words, " \n")
        yield LlamaCompletionResponse(content=words, id_slot=0, stop=True, index=0)

    def is_alive(self) -> bool:
        return self.mock_is_alive

    def props(self) -> LlamaProps:
        return LlamaProps(model_name="dummy model")


def get_mocked_backend(timeout: int = 10000) -> LlamaBackend:
    backend = LlamaBackend("localhost:12345", timeout)
    backend.llama = LlamaMock(timeout)  # type: ignore
    return backend


# --------------------------------------------------------------------------------------------------


def test_is_alive() -> None:
    backend = get_mocked_backend()
    backend.llama.mock_is_alive = False  # type: ignore
    assert not backend.is_alive()
    backend.llama.mock_is_alive = True  # type: ignore
    assert backend.is_alive()


def test_model_props() -> None:
    backend = get_mocked_backend()
    mock_model_props = backend.model_props()
    assert mock_model_props.model_name == "dummy model"  # type: ignore


@pytest.mark.anyio
async def test_get_llm_response() -> None:
    backend = get_mocked_backend()
    expected_response = "This is a dummy response"
    expected_chunks = ["This ", "is ", "a ", "dummy ", "response"]
    received_chunks = []

    backend.llama.mock_response = expected_response  # type: ignore
    async for response in backend.get_llm_response(""):
        received_chunks.append(response)

    assert expected_chunks == received_chunks


@pytest.mark.anyio
async def test_get_buffered_llm_response_single_message() -> None:
    backend = get_mocked_backend()
    expected_response = "This is a dummy response"
    expected_chunks = ["This ", "is ", "a ", "dummy ", "response"]
    received_chunks = []
    received_messages = []
    received_response = ""

    backend.llama.mock_response = expected_response  # type: ignore
    async for response_chunk in backend.get_buffered_llm_response("", 100):
        if response_chunk.end_of_message:
            received_messages.append(response_chunk.message)

        if response_chunk.end_of_response:
            received_messages.append(response_chunk.message)
            received_response = response_chunk.response

        if response_chunk.chunk is not None:
            received_chunks.append(response_chunk.chunk)

    assert expected_chunks == received_chunks
    assert len(received_messages) == 1
    assert received_messages[0] == expected_response
    assert received_response == expected_response


@pytest.mark.anyio
async def test_get_buffered_llm_response_single_split_message() -> None:
    backend = get_mocked_backend()
    expected_response = "This is a dummy, but also pretty long response"
    expected_messages = ["This is a dummy, but also", "pretty long response"]
    expected_chunks = ["This ", "is ", "a ", "dummy, ", "but ", "also ", "pretty ", "long ", "response"]
    received_chunks = []
    received_messages = []
    received_response = ""

    backend.llama.mock_response = expected_response  # type: ignore
    async for response_chunk in backend.get_buffered_llm_response("", 30):
        if response_chunk.end_of_message:
            received_messages.append(response_chunk.message)

        if response_chunk.end_of_response:
            received_messages.append(response_chunk.message)
            received_response = response_chunk.response

        if response_chunk.chunk is not None:
            received_chunks.append(response_chunk.chunk)

    assert received_chunks == expected_chunks
    assert received_messages == expected_messages
    assert received_response == expected_response


@pytest.mark.anyio
async def test_get_buffered_llm_response_multiple_split_message() -> None:
    backend = get_mocked_backend()
    expected_response = "This is a dummy, but also pretty long response"
    expected_messages = ["This is a", "dummy, but", "also pretty", "long response"]
    expected_chunks = ["This ", "is ", "a ", "dummy, ", "but ", "also ", "pretty ", "long ", "response"]
    received_chunks = []
    received_messages = []
    received_response = ""

    backend.llama.mock_response = expected_response  # type: ignore
    async for response_chunk in backend.get_buffered_llm_response("", 15):
        if response_chunk.end_of_message:
            received_messages.append(response_chunk.message)

        if response_chunk.end_of_response:
            received_messages.append(response_chunk.message)
            received_response = response_chunk.response

        if response_chunk.chunk is not None:
            received_chunks.append(response_chunk.chunk)

    assert received_chunks == expected_chunks
    assert received_messages == expected_messages
    assert received_response == expected_response


@pytest.mark.anyio
async def test_get_buffered_llm_response_multiline_split_message() -> None:
    backend = get_mocked_backend()
    expected_response = """
This is a dummy, but also pretty long response.
It also contains content separated by newlines.
That's a long message!
Let's see if this thing works properly.
    """.strip()
    expected_messages = [
        "This is a dummy, but also pretty long response.\nIt also contains content separated by newlines.",
        "That's a long message!\nLet's see if this thing works properly.",
    ]
    expected_chunks = [
        "This ",
        "is ",
        "a ",
        "dummy, ",
        "but ",
        "also ",
        "pretty ",
        "long ",
        "response.\n",
        "It ",
        "also ",
        "contains ",
        "content ",
        "separated ",
        "by ",
        "newlines.\n",
        "That's ",
        "a ",
        "long ",
        "message!\n",
        "Let's ",
        "see ",
        "if ",
        "this ",
        "thing ",
        "works ",
        "properly.",
    ]
    received_chunks = []
    received_messages = []
    received_response = ""

    backend.llama.mock_response = expected_response  # type: ignore
    async for response_chunk in backend.get_buffered_llm_response("", 100):
        if response_chunk.end_of_message:
            received_messages.append(response_chunk.message)

        if response_chunk.end_of_response:
            received_messages.append(response_chunk.message)
            received_response = response_chunk.response

        if response_chunk.chunk is not None:
            received_chunks.append(response_chunk.chunk)

    assert received_chunks == expected_chunks
    assert received_messages == expected_messages
    assert received_response == expected_response


@pytest.mark.anyio
async def test_get_buffered_llm_response_codeblock_split_message() -> None:
    backend = get_mocked_backend()
    expected_response = """
This is an example response containing a code block:

```py
def main():
    print("Hello, world!")

if __name__ == '__main__':
    main()
```

Here you go!
    """.strip()
    expected_messages = [
        """
This is an example response containing a code block:

```py
def main():
    print("Hello, world!")

```
        """.strip(),
        """
```py
if __name__ == '__main__':
    main()
```

Here you go!
        """.strip(),
    ]
    received_messages = []
    received_response = ""

    backend.llama.mock_response = expected_response  # type: ignore
    async for response_chunk in backend.get_buffered_llm_response("", 100):
        if response_chunk.end_of_message:
            received_messages.append(response_chunk.message)

        if response_chunk.end_of_response:
            received_messages.append(response_chunk.message)
            received_response = response_chunk.response

    assert received_messages == expected_messages
    assert received_response == expected_response
