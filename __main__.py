"""Main executable module of unreasonable-llama-discord bot"""

import argparse
import logging
import os
import time

import discord
from transformers import AutoTokenizer
from unreasonable_llama import (
    LlamaCompletionRequest,
    UnreasonableLlama,
)

MESSAGE_EDIT_COOLDOWN_MS = 750
MESSAGE_LENGTH_LIMIT = 1990
BOT_PREFIX = "$llm"
SYSTEM_PROMPT = "You are a helpful AI assistant. Help your users with anything they require and explain your thought process."


def current_time_ms() -> int:
    return round(time.time() * 1000)


async def generate_streamed_llm_response(
    llama: UnreasonableLlama, prompt: str, system_prompt: str, tokenizer: AutoTokenizer
) -> LlamaCompletionRequest:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    chat_prompt = str(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    ).removeprefix(tokenizer.bos_token)
    logging.info(
        f"Requesting completion for following formatted prompt:\n{chat_prompt}"
    )

    request = LlamaCompletionRequest(prompt=chat_prompt)
    logging.debug(f"Performing completion request: {request}")
    async for chunk in llama.get_streamed_completion(request):
        logging.debug(f"Got response chunk from LLM: {chunk}")
        yield chunk


def split_message(message: str, threshold: int) -> tuple[str, str | None]:
    threshold = threshold - 5  # safety threshold for adding end markers and stuff
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
            starting_marker = starting_marker + last_marker_slice[3:]

        first_message = first_message + "\n" + ending_marker
        second_message = starting_marker + "\n" + second_message

    return first_message, second_message


async def request_and_process_llm_response(
    message: discord.Message,
    llama: UnreasonableLlama,
    prompt: str,
    system_prompt: str,
    tokenizer: AutoTokenizer,
):
    response_message = None
    buffered_chunks = ""
    time_since_last_update = None

    async for chunk in generate_streamed_llm_response(
        llama, prompt, system_prompt, tokenizer
    ):
        buffered_chunks += chunk.content

        if response_message is None:
            response_message = await message.channel.send(buffered_chunks)
            time_since_last_update = current_time_ms()
            logging.debug(
                f"Created response message, current time: {time_since_last_update}"
            )
            buffered_chunks = ""
        else:
            current_time = current_time_ms()
            logging.debug(
                f"checking cooldown, current time: {current_time}, last time: {time_since_last_update}, diff = {current_time - time_since_last_update}"
            )
            if (
                current_time - time_since_last_update
            ) >= MESSAGE_EDIT_COOLDOWN_MS or chunk.stop:
                new_content = response_message.content + buffered_chunks
                buffered_chunks = ""

                new_content_first, new_content_second = split_message(
                    new_content,
                    MESSAGE_LENGTH_LIMIT,
                )

                response_message = await response_message.edit(
                    content=new_content_first
                )
                if new_content_second is not None:
                    response_message = await response_message.channel.send(
                        new_content_second
                    )

                time_since_last_update = current_time_ms()
                logging.debug(
                    f"Updated message, current time: {time_since_last_update}"
                )


def setup_client(
    client: discord.Client, llama: UnreasonableLlama, tokenizer: AutoTokenizer
):
    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")
        llama_health = llama.get_health(include_slots=True)
        llm_slot = llama_health.slots[0]
        await client.change_presence(
            activity=discord.CustomActivity(f"Currently using {llm_slot.model}")
        )

    @client.event
    async def on_message(message: discord.Message):
        if message.author == client.user:
            return

        if message.content.startswith(BOT_PREFIX):
            prompt = message.content.removeprefix(BOT_PREFIX).strip()
            logging.info(f"Requesting completion for prompt: {prompt}")
            async with message.channel.typing():
                await request_and_process_llm_response(
                    message, llama, prompt, SYSTEM_PROMPT, tokenizer
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path_or_url",
        type=str,
        help="Path or huggingface.co URL to the model for tokenizer purposes.",
    )

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_url)

    llama = UnreasonableLlama()

    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)
    setup_client(client, llama, tokenizer)
    client.run(os.getenv("UNREASONABLE_LLAMA_DISCORD_API_KEY"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
