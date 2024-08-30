"""llama.cpp server support"""

from collections.abc import AsyncIterator
from unreasonable_llama import LlamaCompletionRequest, LlamaSlot, UnreasonableLlama

from dataclasses import dataclass


@dataclass
class LlamaResponseChunk:
    message: str
    """Preprocessed message returned from LLM."""
    chunk: str
    """Raw chunk of LLM response."""
    response: str
    """Unprocessed, full LLM response assembled from chunks."""
    end_of_message: bool
    """When True, current message has maximum length and next response chunk will contain next message."""


def split_message(message: str, threshold: int) -> tuple[str, str | None]:
    threshold = threshold - 5  # safety threshold for ```\n + 1 "to be sure"
    if threshold < 0:
        raise RuntimeError("Threshold cannot be lower than 5!")

    if len(message) < threshold:
        return message, None

    # split the message so first one has up to `threshold` characters,
    # and second one is the remainder
    first_message = message[:threshold]
    second_message = message[threshold:]

    # move everything past last newline from first to second message
    last_newline = first_message.rfind("\n")
    second_message = first_message[last_newline + 1 :] + second_message
    first_message = first_message[:last_newline]

    # check for unclosed code blocks
    # if the number of markers is odd, there's most likely an unclosed code
    # block there
    starting_marker = "```"
    ending_marker = "```"
    if first_message.count(ending_marker) % 2 != 0:
        # find last ``` and check if it has a language marker
        last_marker = first_message.rfind(ending_marker)
        last_marker_ending = first_message.find("\n", last_marker)
        last_marker_slice = first_message[last_marker:last_marker_ending]
        if len(last_marker_slice) > 3:
            starting_marker += last_marker_slice[3:]

        first_message += "\n" + ending_marker
        second_message = starting_marker + "\n" + second_message

    return first_message, second_message


class LlamaBackend:
    def __init__(self, server_url: str, request_timeout: int) -> None:
        self.llama = UnreasonableLlama(server_url, request_timeout)

    def is_alive(self) -> bool:
        return bool(self.llama.is_alive())

    def model_info(self) -> LlamaSlot:
        return self.llama.slots()[0]

    async def get_llm_response(
        self,
        prompt: str,
    ) -> AsyncIterator[str]:
        request = LlamaCompletionRequest(prompt=prompt)
        async for chunk in self.llama.get_streamed_completion(request):
            yield chunk.content

    async def get_buffered_llm_response(self, prompt: str, message_length: int) -> AsyncIterator[LlamaResponseChunk]:
        message = ""
        response = ""

        async for chunk in self.get_llm_response(prompt):
            response += chunk
            message += chunk
            current_message, next_message = split_message(message, message_length)

            if next_message is None:
                yield LlamaResponseChunk(message, chunk, response, False)
            else:
                yield LlamaResponseChunk(message, chunk, response, True)
                yield LlamaResponseChunk(next_message, chunk, response, False)
                message = next_message
