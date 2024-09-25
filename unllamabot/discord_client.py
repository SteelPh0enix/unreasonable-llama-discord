"""Utilities for interacting with Discord"""

from __future__ import annotations

import logging
from pathlib import Path

import discord
from bot_config import SteelLlamaConfig
from bot_database import BotDatabase, ChatRole
from llama_backend import LlamaBackend
from llm_utils import LLMUtils


class SteelLlamaDiscordClient(discord.Client):
    def __init__(self, config: SteelLlamaConfig, llm_utils: LLMUtils) -> None:
        self.config = config
        self.llm_utils = llm_utils
        self.db = BotDatabase(config.chat_database_path, config.default_system_prompt)
        self.backend = LlamaBackend(
            config.llama_url if config.llama_url is not None else "",
            config.llama_request_timeout,
        )

        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

    async def on_ready(self) -> None:
        if not self.backend.is_alive():
            raise RuntimeError("Backend is not running or configured IP is invalid!")

        model_props = self.backend.model_props()
        model_name = Path(model_props.default_generation_settings.model).name
        model_context_length = model_props.default_generation_settings.n_ctx
        logging.debug(f"Loaded model: {model_name}")

        await self.change_presence(
            activity=discord.CustomActivity(
                f"Chat with me using {self.config.bot_prefix}{self.config.commands['inference']}! "
                f"Currently using {model_name} with context of {model_context_length} tokens per user."
            )
        )

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

        # messages = self.db.get_user_messages(user_id)
        # completion_message = self.llm_utils.format_messages_into_chat(messages)
        self.db.add_message(user_id, ChatRole.USER, prompt)

    async def process_help_command(self, message: discord.Message, subject: str | None = None) -> None:
        await message.reply("this will be help when i finish it")

    async def process_reset_conversation_command(self, message: discord.Message) -> None:
        pass

    async def process_stats_command(self, message: discord.Message) -> None:
        pass

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

        if command_name == self.config.commands["inference"]:
            await self.process_inference_command(message, arguments)
        elif command_name == self.config.commands["help"]:
            await self.process_help_command(message, arguments)
        elif command_name == self.config.commands["reset-conversation"]:
            await self.process_reset_conversation_command(message)
        elif command_name == self.config.commands["stats"]:
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
