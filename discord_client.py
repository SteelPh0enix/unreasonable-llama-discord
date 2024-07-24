"""Utilities for interacting with Discord"""

import logging

import discord


class SteelLlamaDiscordClient(discord.Client):
    def __init__(self, config: dict):
        self.message_edit_cooldown = config["messages"]["edit-cooldown-ms"]
        self.message_length_limit = config["messages"]["length-limit"]
        self.message_removal_reaction = config["messages"]["remove-reaction"]

        self.bot_prefix = config["commands"]["prefix"]
        self.commands = {
            command_name.removesuffix("-cmd"): command
            for command_name, command in config["commands"].items()
            if command_name.endswith("-cmd")
        }

        self.default_system_prompt = config["bot"]["default-system-prompt"]

        logging.info(f"""Loaded bot configuration:
                     message_edit_cooldown: {self.message_edit_cooldown}
                     message_length_limit: {self.message_length_limit}
                     message_removal_reaction: {self.message_removal_reaction}
                     bot_prefix: {self.bot_prefix}
                     commands: {self.commands}
                     default_system_prompt: {self.default_system_prompt}""")

        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

    async def on_ready(self):
        logging.info("Bot is ready!")

    async def on_message(self, message: discord.Message):
        logging.debug(f"Message detected: {message}")

    def is_reaction_to_be_handled(
        self,
        event: discord.RawReactionActionEvent,
    ) -> bool:
        message_is_from_bot = event.message_author_id == self.user.id
        reaction_was_added_by_bot = event.user_id == self.user.id
        return message_is_from_bot and not reaction_was_added_by_bot

    async def on_raw_reaction_add(self, event: discord.RawReactionActionEvent):
        if not self.is_reaction_to_be_handled(event):
            return

        if str(event.emoji) == self.message_removal_reaction:
            message_channel = await self.fetch_channel(event.channel_id)
            message_to_delete = await message_channel.fetch_message(event.message_id)
            logging.info(f"Removing message {message_to_delete.id}")
            await message_to_delete.delete()
