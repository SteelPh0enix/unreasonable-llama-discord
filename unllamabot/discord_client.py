"""Utilities for interacting with Discord"""

from __future__ import annotations

import logging

import discord
from bot_config import SteelLlamaConfig


class SteelLlamaDiscordClient(discord.Client):
    def __init__(self, config: SteelLlamaConfig):
        self.config = config
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
