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
BOT_REMOVE_MESSAGE_REACTION = "💀"
BOT_PREFIX = "$"
BOT_LLM_INFERENCE_COMMAND = "llm"
BOT_HELP_COMMAND = "llm-help"
BOT_RESET_CONVERSATION_HISTORY_COMMAND = "llm-reset"
BOT_STATS_COMMAND = "llm-stats"
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Help your users with anything they require and explain your thought process."
BOT_HELP_MESSAGE = f"""This is SteelLlama, an [`unreasonable-llama-discord`](https://github.com/SteelPh0enix/unreasonable-llama-discord)-based Discord bot.
It's main functionality is to be a bridge between Discord and locally ran LLM. And by "locally", i mean on admin's GPU somewhere in eastern Poland.
# Available commands:
* `{BOT_PREFIX}{BOT_LLM_INFERENCE_COMMAND} [prompt]` - Give a prompt to an LLM and trigger it's response. Bot will retain (some) conversation history (depending on currently ran LLMs context length and bot's configuration).
* `{BOT_PREFIX}{BOT_HELP_COMMAND}` - Shows this message.
* `{BOT_PREFIX}{BOT_RESET_CONVERSATION_HISTORY_COMMAND}` - Reset your conversation history. This will create a new session with the LLM.
* `{BOT_PREFIX}{BOT_STATS_COMMAND}` - Show your conversation statistics.

Default system prompt: {DEFAULT_SYSTEM_PROMPT}

## How does it's memory work?

**Every user has it's own conversation history.** Bot will save all the prompts and LLM responses and automatically track conversation.
**Bot will NOT track any other messages, only explicit prompts! This is for privacy reasons.**
It's recommended to let the bot finish it's response before requesting a new one. Otherwise, conversation layout may get bugged.
Tracking is based on user ID only, so the bot will use the same conversation history for all servers (and DMs).
In other words - if you start talking to the bot in DMs, you can continue talking to him with the same history on any server (and vice-versa).
**Currently, there's no history trimming implemented (to fit in LLMs context), so it must be cleared manually by the user when it becomes too long.**

Add the {BOT_REMOVE_MESSAGE_REACTION} reaction to a bot's message to remove it. Removing the message this way will not remove it from conversation history."""


def update_help_message(new_suffix: str):
    global BOT_HELP_MESSAGE
    BOT_HELP_MESSAGE = BOT_HELP_MESSAGE + "\n" + new_suffix


def current_time_ms() -> int:
    return round(time.time() * 1000)


type LLMConversation = list[dict[str, str]]


class LLMConversationHistory:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        system_prompt: str,
        user_role_name: str = "user",
        system_role_name: str = "system",
        assistant_role_name: str = "assistant",
    ):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.user_role_name = user_role_name
        self.system_role_name = system_role_name
        self.assistant_role_name = assistant_role_name
        self.clean()

    def _generate_init_prompt(self) -> LLMConversation:
        return [{"role": self.system_role_name, "content": self.system_prompt}]

    def add_user_prompt(self, prompt: str):
        self.conversation_history.append(
            {"role": self.user_role_name, "content": prompt}
        )

    def add_assistant_message(self, message: str):
        self.conversation_history.append(
            {"role": self.assistant_role_name, "content": message}
        )

    def clean(self):
        self.conversation_history = self._generate_init_prompt()

    def generate_prompt(self) -> str:
        """Returns LLM prompt from conversation history"""
        return str(
            self.tokenizer.apply_chat_template(
                self.conversation_history, add_generation_prompt=True, tokenize=False
            )
        ).removeprefix(self.tokenizer.bos_token)

    def conversation_length(self) -> int:
        """Returns conversation length in the amount of tokens"""
        return len(self.tokenizer.tokenize(self.generate_prompt()))


type LLMConversationDatabase = dict[str, LLMConversationHistory]


async def generate_streamed_llm_response(
    llama: UnreasonableLlama, prompt: str
) -> LlamaCompletionRequest:
    logging.info(f"Requesting completion for following formatted prompt:\n{prompt}")
    request = LlamaCompletionRequest(prompt=prompt)
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
    conversation: LLMConversationHistory,
    suffix_token: str,
    message_edit_cooldown_ms: int = MESSAGE_EDIT_COOLDOWN_MS,
    message_length_limit: int = MESSAGE_LENGTH_LIMIT,
):
    prompt = conversation.generate_prompt()
    response_message = None
    response_text = ""
    response_flush_index = 0
    time_since_last_update = None

    async for chunk in generate_streamed_llm_response(llama, prompt):
        response_text += chunk.content
        response_text = response_text.removesuffix(suffix_token)

        if len(response_text.strip()) == 0 and response_message is None:
            # this handles the case where LLM generated empty first message, for example when
            # tokenizer doesn't add newline at the end of message
            continue

        if response_message is None:
            response_message = await message.reply(response_text)
            time_since_last_update = current_time_ms()
            logging.debug(
                f"Created response message, current time: {time_since_last_update}"
            )
            response_flush_index = len(response_text)
        else:
            current_time = current_time_ms()
            logging.debug(
                f"checking cooldown, current time: {current_time}, last time: {time_since_last_update}, diff = {current_time - time_since_last_update}"
            )
            if (
                current_time - time_since_last_update
            ) >= message_edit_cooldown_ms or chunk.stop:
                new_content = (
                    response_message.content + response_text[response_flush_index:]
                )
                response_flush_index = len(response_text)

                new_content_first, new_content_second = split_message(
                    new_content,
                    message_length_limit,
                )

                response_message = await response_message.edit(
                    content=new_content_first
                )
                if new_content_second is not None:
                    response_message = await response_message.reply(new_content_second)

                time_since_last_update = current_time_ms()
                logging.debug(
                    f"Updated message, current time: {time_since_last_update}"
                )

    logging.info(
        f"Finished generating assistant message, saving it into conversation...\n{response_text}"
    )
    conversation.add_assistant_message(response_text)


def setup_client(
    client: discord.Client,
    llama: UnreasonableLlama,
    tokenizer: AutoTokenizer,
    conversations: dict[str, LLMConversationHistory],
    suffix_token: str,
):
    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")
        llama_health = llama.get_health(include_slots=True)
        llm_slot = llama_health.slots[0]
        update_help_message(f"""## Loaded model details
Name: `{llm_slot.model}`
Context length (tokens): `{llm_slot.n_ctx}`
Seed: `{llm_slot.seed}`
Temperature (randomness of generated text, in `(0, 1)` range): `{llm_slot.temperature}`
Dynamic temperature range (final temperature is in range `T +/- dynT`): `{llm_slot.dynatemp_range}`
Dynamic temperature exponent: `{llm_slot.dynatemp_exponent}`
Top K (limits next token selection to K most probable tokens): `{llm_slot.top_k}`
Top P (limits next token selection to a subset of tokens with cumulative probability above P): `{llm_slot.top_p}`
Min P (minimum probability for a token to be considered, relative to prob. of most likely token): `{llm_slot.min_p}`
Samplers: `{llm_slot.samplers}`""")
        await client.change_presence(
            activity=discord.CustomActivity(
                f"Try me with {BOT_PREFIX}{BOT_HELP_COMMAND}. Currently using {llm_slot.model} with {llm_slot.n_ctx} context length"
            )
        )

    @client.event
    async def on_message(message: discord.Message):
        # ignore bot messages
        if message.author == client.user:
            return

        # ignore non-commands
        if not message.content.startswith(BOT_PREFIX):
            return

        message_author = message.author.id
        raw_message_text = message.content.removeprefix(BOT_PREFIX).strip()
        message_split = raw_message_text.split(" ", 1)
        if len(message_split) == 1:
            command = message_split[0]
            argument = None
        else:
            command, argument = message_split
        deletable_replies: list[discord.Message] = []

        if command == BOT_LLM_INFERENCE_COMMAND:
            logging.info(
                f"Requesting completion for prompt: {argument} from user {message_author}"
            )

            if message_author not in conversations:
                logging.info(
                    f"Creating new conversation history for user {message_author}"
                )
                conversations[message_author] = LLMConversationHistory(
                    tokenizer, DEFAULT_SYSTEM_PROMPT
                )

            conversations[message_author].add_user_prompt(argument)
            conversation_length = conversations[message_author].conversation_length()
            logging.info(
                f"User's {message_author} conversation history is {conversation_length} tokens long"
            )

            async with message.channel.typing():
                await request_and_process_llm_response(
                    message,
                    llama,
                    conversations[message_author],
                    suffix_token,
                )

        elif command == BOT_HELP_COMMAND:
            logging.info("Help requested")
            async with message.channel.typing():
                help_first, help_second = split_message(
                    BOT_HELP_MESSAGE, MESSAGE_LENGTH_LIMIT
                )
                deletable_replies.append(await message.reply(help_first))
                if help_second is not None:
                    deletable_replies.append(await message.reply(help_second))

        elif command == BOT_RESET_CONVERSATION_HISTORY_COMMAND:
            logging.info(f"Removing conversation history for {message_author}...")
            if message_author in conversations:
                del conversations[message_author]
                logging.info(f"Conversation history for {message_author} deleted!")
                deletable_replies.append(
                    await message.reply("Conversation history cleared!")
                )
            else:
                logging.info(
                    f"There's no conversation history saved for {message_author}"
                )
                deletable_replies.append(
                    await message.reply("There's no conversation history to clear!")
                )

        elif command == BOT_STATS_COMMAND:
            logging.info(f"Stats for user {message_author} requested!")
            if message_author in conversations:
                convo = conversations[message_author]
                context_length = convo.conversation_length()
                messages_in_convo = len(convo.conversation_history)
                system_prompt = convo.conversation_history[0]["content"]
                deletable_replies.append(
                    await message.reply(
                        f"User ID: {message_author}\nMessages in conversation history: {messages_in_convo}\nMessage history context length: {context_length}\nSystem prompt: {system_prompt}"
                    )
                )
            else:
                deletable_replies.append(
                    await message.reply(
                        f"No statistics stored for user {message_author}."
                    )
                )

        for reply in deletable_replies:
            await reply.add_reaction(BOT_REMOVE_MESSAGE_REACTION)

    @client.event
    async def on_raw_reaction_add(event: discord.RawReactionActionEvent):
        logging.debug(f"Client user: {client.user.id}, reaction event: {event}")
        # ignore non-bot messages
        if event.message_author_id != client.user.id:
            return

        # ignore event of adding the reaction by bot
        if event.user_id == client.user.id:
            return

        # ignore other emojis
        if str(event.emoji) != BOT_REMOVE_MESSAGE_REACTION:
            return

        message_channel = await client.fetch_channel(event.channel_id)
        message_to_delete = await message_channel.fetch_message(event.message_id)
        logging.info(f"Removing message {message_to_delete.id}")
        await message_to_delete.delete()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path_or_url",
        type=str,
        help="Path or huggingface.co URL to the model for tokenizer purposes.",
    )

    parser.add_argument(
        "--suffix-token", type=str, help="Custom suffix token to trim from messages"
    )

    args = parser.parse_args()

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    llama = UnreasonableLlama()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path_or_url, trust_remote_code=True
    )
    suffix_token = (
        args.suffix_token if args.suffix_token is not None else tokenizer.eos_token
    )
    logging.info(f"Suffix tokens: {suffix_token}")
    conversations: dict[str, LLMConversationHistory] = {}

    setup_client(client, llama, tokenizer, conversations, suffix_token)
    client.run(os.getenv("UNREASONABLE_LLAMA_DISCORD_API_KEY"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
