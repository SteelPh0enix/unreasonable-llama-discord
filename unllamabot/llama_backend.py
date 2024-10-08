"""llama.cpp server support"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import cast

from unreasonable_llama import LlamaCompletionRequest, LlamaProps, LlamaTokenizeRequest, UnreasonableLlama


@dataclass
class LlamaResponseChunk:
    message: str
    """Preprocessed message returned from LLM."""
    chunk: str | None
    """Raw chunk of LLM response, will be None in last chunk of current message."""
    response: str
    """Unprocessed, full LLM response assembled from chunks."""
    end_of_message: bool
    """If True, current chunk is the last chunk in current message"""
    end_of_response: bool
    """If True, current chunk is the last chunk generated by LLM"""
    new_message: bool
    """If True, previous message has reached maximum length and current response chunk contains new message. False in first chunk."""


def find_last_occurence(input: str, strings: list[str]) -> tuple[int, str] | None:
    for s in strings:
        found_char_index = input.rfind(s)
        if found_char_index > 0:
            return found_char_index, s
    return None


def split_message(message: str, threshold: int) -> tuple[str, str | None]:
    """Splits the message into two at a character threshold, preserving code blocks.
    Tries to split it smart, on newline/word/sentence boundary."""
    if len(message) < threshold:
        return message, None

    # do a cheap hard split first
    first_message = message[:threshold]
    second_message = message[threshold:]

    # now, we need to check if split can be done better.
    # we want to avoid splitting words, and preferably - sentences.
    # to do that, we look for the last substring that follows the thing in the
    # first message, and split at this boundary instead, by moving the context
    # to the second message.
    # the priority is following: paragraph ('\n'), sentence ('.'), word (' ').
    if occurence := find_last_occurence(first_message, ["\n", ". ", " "]):
        split_position, separator = occurence
        # copy the context from first message, post-split, to second message
        second_message = first_message[split_position + len(separator) :] + second_message
        # shorten the first message, preserving the visible part of separator
        first_message = first_message[
            : split_position if separator.isspace() else split_position + len(separator.rstrip())
        ]

    # check for unclosed code blocks
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
        return self.llama.is_alive()

    def model_props(self) -> LlamaProps:
        return self.llama.props()

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
                yield LlamaResponseChunk(
                    message=current_message,
                    chunk=chunk.content,
                    response=response,
                    end_of_message=False,
                    end_of_response=chunk.stop,
                    new_message=False,
                )
            else:
                yield LlamaResponseChunk(
                    message=current_message,
                    chunk=None,
                    response=response,
                    end_of_message=True,
                    end_of_response=False,
                    new_message=False,
                )
                yield LlamaResponseChunk(
                    message=next_message,
                    chunk=chunk.content,
                    response=response,
                    end_of_message=False,
                    end_of_response=chunk.stop,
                    new_message=True,
                )
                message = next_message

    def tokenize(self, message: str) -> list[int]:
        request = LlamaTokenizeRequest(message)
        return cast(list[int], self.llama.tokenize(request))
