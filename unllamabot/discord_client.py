"""Utilities for interacting with Discord"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import discord
from bot_config import BotConfig
from bot_database import BotDatabase, ChatRole
from llama_backend import LlamaBackend
from llm_utils import LLMUtils


def current_time_ms() -> int:
    return round(time.time() * 1000)


class SteelLlamaDiscordClient(discord.Client):
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.db = BotDatabase(config.chat_database_path, config.default_system_prompt)
        self.backend = LlamaBackend(
            config.llama_url if config.llama_url is not None else "",
            config.llama_request_timeout,
        )
        self.llm_utils = LLMUtils(self.backend)

        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

    async def on_ready(self) -> None:
        if not self.backend.is_alive():
            raise RuntimeError("Backend is not running or configured IP is invalid!")

        model_props = self.backend.model_props()
        model_name = Path(model_props.default_generation_settings.model).name
        model_context_length = model_props.default_generation_settings.n_ctx
        logging.info(f"Loaded model: {model_name}")

        presence_string = (
            f"Chat with me using {self.config.bot_prefix}{self.config.commands['inference']}! "
            + f"Currently using {model_name} with context of {model_context_length} tokens per user."
        )
        logging.info(f"Bot presence: {presence_string}")
        await self.change_presence(activity=discord.CustomActivity(presence_string))
        logging.info("Bot is ready!")

    async def process_inference_command(self, message: discord.Message, prompt: str | None) -> None:
        if prompt is None:
            prefixed_inference_command = f"{self.config.bot_prefix}{self.config.commands['inference']}"
            prefixed_help_command = f"{self.config.bot_prefix}{self.config.commands['help']}"
            await message.reply(
                f"*Usage: `{prefixed_inference_command} [message]`, for example `{prefixed_inference_command} what's the highest mountain on earth?`*\n"
                f"*Use `{prefixed_help_command}` for details about the bot commands.*"
            )
            return

        user_id = message.author.id
        user = self.db.get_or_create_user(user_id)
        if not self.db.user_has_messages(user_id):
            self.db.add_message(user_id, ChatRole.SYSTEM, user.system_prompt)

        self.db.add_message(user_id, ChatRole.USER, prompt)
        messages = self.db.get_user_messages(user_id)
        llm_prompt = self.llm_utils.format_messages_into_chat(messages)

        # This will be a placeholder, LLM windup can take a while.
        reply_message = await message.reply("*Generating response, please wait...*")
        last_update_time = current_time_ms()
        full_response = None

        logging.debug(f"Processing inference command for user {user_id}...")
        logging.debug(f"LLM prompt: {llm_prompt}")

        async for chunk in self.backend.get_buffered_llm_response(llm_prompt, self.config.message_length_limit):
            logging.debug(f"Received response chunk: {chunk}")
            if chunk.new_message:
                reply_message = await reply_message.reply(content=chunk.message)
            else:
                if current_time_ms() - last_update_time >= self.config.message_edit_cooldown:
                    reply_message = await reply_message.edit(content=chunk.message)
                    last_update_time = current_time_ms()

            if chunk.end_of_response:
                reply_message = await reply_message.edit(content=chunk.message)
                last_update_time = current_time_ms()
                full_response = chunk.response

        self.db.add_message(user_id, ChatRole.BOT, full_response)

    async def process_help_command(self, message: discord.Message, subject: str | None = None) -> None:
        help_content = f"*No help available for selected subject. Try {self.config.bot_prefix}{self.config.commands['help']} for list of subjects and generic help.*"
        match subject:
            case None:
                help_content = f"""# This is [UnreasonableLlama](https://pypi.org/project/unreasonable-llama/)-based Discord bot.
It allows you to converse with an LLM hosted via llama.cpp.
The bot remembers your conversations and allows you to configure the LLM in some degree.

[This bot is open-source](https://github.com/SteelPh0enix/unreasonable-llama-discord).
## Available commands:
    * `{self.config.bot_prefix}{self.config.commands["inference"].command} [message]` - chat with the LLM
    * `{self.config.bot_prefix}{self.config.commands["help"].command} [subject (optional)]` - show help
    * `{self.config.bot_prefix}{self.config.commands["reset-conversation"].command}` - clear your conversation history and start a new one
    * `{self.config.bot_prefix}{self.config.commands["stats"].command}` - show some stats of your conversation
## Available help subjects:
    * `model` - show model details
    * `admin` - show admin commands"""
            case "model":
                props = self.backend.model_props()
                model_info = props.default_generation_settings
                help_content = f"""# Currently loaded model: {Path(model_info.model).name}
**Context length**: {model_info.n_ctx} tokens
**Samplers**: {model_info.samplers}
## Parameters
**Top-K**: {model_info.top_k}
**Tail-Free Sampling Z**: {model_info.tfs_z:.02f}
**Typical-P**: {model_info.typical_p:.02f}
**Top-P**: {model_info.top_p:.02f}
**Min-P**: {model_info.min_p:.02f}
**Temperature**: {model_info.temperature:.02f}
**Mirostat type**: {model_info.mirostat}
**Mirostat learning rate (Eta)**: {model_info.mirostat_eta:.02f}
**Mirostat target entropy (Tau)**: {model_info.mirostat_tau:.02f}"""
            case "admin":
                help_content = f"""# Admin commands
* `{self.config.bot_prefix}{self.config.commands["refresh"].command}` - refresh llama.cpp props"""

        await message.reply(content=help_content)

    async def process_reset_conversation_command(self, message: discord.Message) -> None:
        self.db.clear_user_messages(message.author.id)
        await message.reply("Message history cleared!")

    async def process_stats_command(self, message: discord.Message) -> None:
        user_id = message.author.id
        messages = self.db.get_user_messages(user_id)
        if len(messages) == 0:
            await message.reply("Your chat history doesn't exist!")
            return

        llm_prompt = self.llm_utils.format_messages_into_chat(messages)
        tokenized_prompt = self.backend.tokenize(llm_prompt)
        context_length = self.backend.model_props().default_generation_settings.n_ctx
        prompt_length_tokens = len(tokenized_prompt)
        prompt_length_chars = len(llm_prompt)
        context_percent_used = (prompt_length_tokens / context_length) * 100
        await message.reply(
            f"""Messages in chat history (including system prompt): {len(messages)}
Current prompt length (tokens): {prompt_length_tokens}/{context_length} ({context_percent_used:.2f}% used)
Current prompt length (characters): {prompt_length_chars}"""
        )

    async def on_message(self, message: discord.Message) -> None:
        # ignore your own messages
        if message.author == self.user:
            return

        # ignore messages without prefix
        if not message.content.startswith(self.config.bot_prefix):
            return

        command_parts = message.content.lstrip(self.config.bot_prefix).split(" ", 1)
        command_name = command_parts[0]
        arguments = command_parts[1] if len(command_parts) > 1 else None

        logging.info(
            f"<UID:{message.author.id}|UN:{message.author.global_name}> Command detected: {command_name}, arguments: {arguments}"
        )

        if command_name == self.config.commands["inference"].command:
            await self.process_inference_command(message, arguments)
        elif command_name == self.config.commands["help"].command:
            await self.process_help_command(message, arguments)
        elif command_name == self.config.commands["reset-conversation"].command:
            await self.process_reset_conversation_command(message)
        elif command_name == self.config.commands["stats"].command:
            await self.process_stats_command(message)
        else:
            await message.reply(f"Unknown command: {command_name}")

    def should_reaction_be_handled(
        self,
        event: discord.RawReactionActionEvent,
    ) -> bool:
        if user := self.user:
            message_is_from_bot = event.message_author_id == user.id
            reaction_was_added_by_bot = event.user_id == user.id
            return message_is_from_bot and not reaction_was_added_by_bot
        return False

    async def on_raw_reaction_add(self, event: discord.RawReactionActionEvent) -> None:
        if not self.should_reaction_be_handled(event):
            return

        if str(event.emoji) == self.config.message_removal_reaction:
            message_channel = await self.fetch_channel(event.channel_id)
            if isinstance(
                message_channel,
                discord.TextChannel | discord.Thread,
            ):
                message_to_delete = await message_channel.fetch_message(event.message_id)
                logging.info(f"Removing message {message_to_delete.id}")
                await message_to_delete.delete()
            else:
                logging.warning(
                    f"Message removal emoji received from channel {message_channel} - cannot fetch and delete the target message!"
                )
