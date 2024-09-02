"""llama.cpp server support"""

from collections.abc import AsyncIterator
from unreasonable_llama import LlamaCompletionRequest, LlamaSlot, UnreasonableLlama

from dataclasses import dataclass


@dataclass
class LlamaResponseChunk:
    message: str
    """Preprocessed message returned from LLM."""
    chunk: str | None
    """Raw chunk of LLM response, will be None in terminator chunk (end_of_message == True)"""
    response: str
    """Unprocessed, full LLM response assembled from chunks."""
    end_of_message: bool
    """When True, current message has maximum length and next response chunk will contain next message."""


def find_last_occurence_of_character(string: str, characters: str) -> int | None:
    for character in characters:
        found_char_index = string.rfind(character)
        if found_char_index > 0:
            return found_char_index
    return None


def split_message(message: str, threshold: int) -> tuple[str, str | None]:
    """Splits the message into two at threshold, preserving code blocks.
    Tries to split it smart, on newline/word boundary."""
    if len(message) < threshold:
        return message, None

    first_message = message[:threshold]
    second_message = message[threshold:]
    if split_position := find_last_occurence_of_character(first_message, "\n "):
        second_message = first_message[split_position + 1 :] + second_message
        first_message = first_message[:split_position]

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
        # if it does, add it to starting marker
        if len(last_marker_slice) > 3:
            starting_marker += last_marker_slice[3:]

        # close block in first message, open in second and put the code there
        first_message += f"\n{ending_marker}"
        second_message = f"{starting_marker}\n{second_message}"

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
        request = LlamaCompletionRequest(prompt=prompt)
        message = ""
        response = ""

        async for chunk in self.llama.get_streamed_completion(request):
            response += chunk.content
            message += chunk.content
            current_message, next_message = split_message(message, message_length)

            if next_message is None:
                yield LlamaResponseChunk(message, chunk.content, response, chunk.stop)
            else:
                yield LlamaResponseChunk(current_message, None, response, True)
                yield LlamaResponseChunk(next_message, chunk.content, response, chunk.stop)
                message = next_message
