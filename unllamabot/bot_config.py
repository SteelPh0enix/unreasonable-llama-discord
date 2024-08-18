"""Bot's configuration utilities"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

DEFAULT_CONFIG = """[messages]
edit-cooldown-ms = 750
length-limit = 1990
remove-reaction = "💀"

[commands]
prefix = "$"
inference-cmd = "llm"
help-cmd = "llm-help"
reset-conversation-cmd = "llm-reset"
stats-cmd = "llm-stats"

[bot]
default-system-prompt = "You are a helpful AI assistant. Assist the user best to your ability."
"""


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
    commands: dict[str, str]
    """Mapping of Discord commands to bot's commands"""

    @staticmethod
    def from_dict(config: dict[str, Any]) -> SteelLlamaConfig:
        message_edit_cooldown = config["messages"]["edit-cooldown-ms"]
        message_length_limit = config["messages"]["length-limit"]
        message_removal_reaction = config["messages"]["remove-reaction"]
        bot_prefix = config["commands"]["prefix"]
        default_system_prompt = config["bot"]["default-system-prompt"]
        commands = {
            command_name.removesuffix("-cmd"): command
            for command_name, command in config["commands"].items()
            if command_name.endswith("-cmd")
        }

        return SteelLlamaConfig(
            message_edit_cooldown,
            message_length_limit,
            message_removal_reaction,
            bot_prefix,
            default_system_prompt,
            commands,
        )

    def __str__(self) -> str:
        return f"""Prefix: {self.bot_prefix}
Default system prompt: {self.default_system_prompt}
Message edit cooldown: {self.message_edit_cooldown}
Message length limit: {self.message_length_limit}
Message removal reaction: {self.message_removal_reaction}"""


def load_bot_configuration(path: Path) -> SteelLlamaConfig | None:
    if not path.exists():
        return None

    with path.open("rb") as config_file:
        logging.info(f"Reading configuration from {path}...")
        config_json = tomllib.load(config_file)
        logging.debug(f"Read configuration JSON: {config_json}")
        return SteelLlamaConfig.from_dict(config_json)


def create_default_bot_configuration(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        logging.critical(f"Configuration file {path} exists, and i'm forbidden from overwriting it! Exiting...")
        sys.exit(1)

    with path.open("wb") as config_file:
        config_file.write(DEFAULT_CONFIG.encode("utf-8"))
        logging.info(f"Default config created at {path}.")
