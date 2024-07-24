"""Utilities for interacting with Discord"""

import discord

import logging


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

    async def on_raw_reaction_add(event: discord.RawReactionActionEvent):
        pass
