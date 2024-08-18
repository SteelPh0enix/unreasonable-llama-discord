"""Utilities for interacting with Discord"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import discord


@dataclass
class SteelLlamaConfig:
    message_edit_cooldown: int
    """Time between consecutive message edits, in milliseconds."""
    message_length_limit: int
    """Length limit of a single message. Longer messages will be split into multiple shorter messages."""
    message_removal_reaction: str
    """Reaction to use in order to make the bot remove it's message on Discord"""
    bot_prefix: str
    """Prefix character (or string) of the bot"""
    default_system_prompt: str
    """Default system prompt for the LLM"""

    @staticmethod
    def from_dict(config: dict[str, Any]) -> SteelLlamaConfig:
        message_edit_cooldown = config["messages"]["edit-cooldown-ms"]
        message_length_limit = config["messages"]["length-limit"]
        message_removal_reaction = config["messages"]["remove-reaction"]
        bot_prefix = config["commands"]["prefix"]
        default_system_prompt = config["bot"]["default-system-prompt"]

        return SteelLlamaConfig(
            message_edit_cooldown,
            message_length_limit,
            message_removal_reaction,
            bot_prefix,
            default_system_prompt,
        )

    def __str__(self) -> str:
        return f"""Prefix: {self.bot_prefix}
Default system prompt: {self.default_system_prompt}
Message edit cooldown: {self.message_edit_cooldown}
Message length limit: {self.message_length_limit}
Message removal reaction: {self.message_removal_reaction}"""


class SteelLlamaDiscordClient(discord.Client):
    def __init__(self, config: dict[str, Any]):
        self.config = SteelLlamaConfig.from_dict(config)
        logging.info(f"Loaded bot configuration:\n{self.config}")

        self.commands = {
            command_name.removesuffix("-cmd"): command
            for command_name, command in config["commands"].items()
            if command_name.endswith("-cmd")
        }

        logging.info(f"Loaded commands: {[name for name in self.commands.keys()]}")

        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

    async def on_ready(self) -> None:
        logging.info("Bot is ready!")

    async def on_message(self, message: discord.Message) -> None:
        logging.debug(f"Message detected: {message}")

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
