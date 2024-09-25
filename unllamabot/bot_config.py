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

[commands.inference]
cmd = "llm"
requires_admin = false

[commands.help]
cmd = "llm-help"
requires_admin = false

[commands.reset-conversation]
cmd = "llm-reset"
requires_admin = false

[commands.stats]
cmd = "llm-stats"
requires_admin = false

[commands.refresh]
cmd = "llm-refresh"
requires_admin = true

[bot]
prefix = "$"
default-system-prompt = "You are a helpful AI assistant. Assist the user best to your ability."
chat-database-path = "./chats.uldb"
# add user IDs of bot administrators here, those users can call admin commands.
admins-id = []

[llama]
# URL is optional, if not provided it will be loaded from LLAMA_CPP_SERVER_URL env variable,
# as it's UnreasonableLlama fallback behaviour.
# url = "http://127.0.0.1:8080/"
server_timeout = 10000
"""


@dataclass(frozen=True)
class BotCommand:
    command: str
    """Command string."""
    requires_admin: bool
    """If True, command can be executed only by user with ID from admins-id list."""


@dataclass(frozen=True)
class BotConfig:
    message_edit_cooldown: int
    """Time between consecutive message edits, in milliseconds."""
    message_length_limit: int
    """Length limit of a single message. Longer messages will be split into multiple shorter messages."""
    message_removal_reaction: str
    """Reaction to use in order to make the bot remove it's message on Discord."""
    bot_prefix: str
    """Prefix character (or string) of the bot."""
    default_system_prompt: str
    """Default system prompt for the LLM."""
    admins_id: list[int]
    """List bot administrators user IDs"""
    commands: dict[str, BotCommand]
    """List of commands."""
    llama_request_timeout: int
    """Timeout for llama.cpp server requests, in milliseconds."""
    llama_url: str | None
    """llama.cpp server URL, if None the LLAMA_CPP_SERVER_URL env var will be used instead."""
    chat_database_path: str
    """Path to bot's chats and users database."""

    @staticmethod
    def from_dict(config: dict[str, Any]) -> BotConfig:
        message_edit_cooldown = config["messages"]["edit-cooldown-ms"]
        message_length_limit = config["messages"]["length-limit"]
        message_removal_reaction = config["messages"]["remove-reaction"]
        commands = {
            command_name: BotCommand(command_info["cmd"], command_info["requires_admin"])
            for command_name, command_info in config["commands"].items()
        }
        bot_prefix = config["bot"]["prefix"]
        default_system_prompt = config["bot"]["default-system-prompt"]
        chat_database_path = config["bot"]["chat-database-path"]
        admin_ids = config["bot"]["admins-id"]
        llama_request_timeout = config["llama"]["server_timeout"]
        llama_url = config["llama"].get("url")

        return BotConfig(
            message_edit_cooldown,
            message_length_limit,
            message_removal_reaction,
            bot_prefix,
            default_system_prompt,
            admin_ids,
            commands,
            llama_request_timeout,
            llama_url,
            chat_database_path,
        )


def load_bot_configuration(path: Path) -> BotConfig | None:
    if not path.exists():
        return None

    with path.open("rb") as config_file:
        logging.info(f"Reading configuration from {path}...")
        config_json = tomllib.load(config_file)
        logging.debug(f"Read configuration JSON: {config_json}")
        return BotConfig.from_dict(config_json)


def create_default_bot_configuration(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        logging.critical(f"Configuration file {path} exists, and i'm forbidden from overwriting it! Exiting...")
        sys.exit(1)

    with path.open("wb") as config_file:
        config_file.write(DEFAULT_CONFIG.encode("utf-8"))
        logging.info(f"Default config created at {path}.")
