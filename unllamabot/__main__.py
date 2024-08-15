"""Entry point for unreasonable-llama-discord bot"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import tomllib

from discord_client import SteelLlamaDiscordClient

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


def load_bot_configuration(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("rb") as config_file:
        logging.info(f"Reading configuration from {path}...")
        return tomllib.load(config_file)


def create_default_bot_configuration(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        logging.critical(
            f"Configuration file {path} exists, and i'm forbidden from overwriting it! Exiting..."
        )
        sys.exit(1)

    with path.open("wb") as config_file:
        config_file.write(DEFAULT_CONFIG.encode("utf-8"))
        logging.info(f"Default config created at {path}.")


def parse_script_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unreasonable Llama Discord bot")

    parser.add_argument(
        "--config-file",
        type=str,
        default="unreasonable-config.toml",
        help="Path to bot's configuration file. "
        "If the file doesn't exist, then it will be created with default configuration.",
    )

    parser.add_argument(
        "--overwrite-config-file",
        action="store_true",
        default=False,
        help="When used, config file will be overwritten if it exists but cannot be loaded. "
        "Default behavior is exitting with exit code 1 instead.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical", "none"],
        default="info",
        help="Log level, 'info' by default",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    config_path = Path(args.config_file)
    bot_config = load_bot_configuration(config_path)
    if bot_config is None:
        logging.debug("Could not load configuration, creating default...")
        create_default_bot_configuration(config_path, args.overwrite_config_file)
        bot_config = load_bot_configuration(config_path)

    logging.debug(f"Loaded configuration: {bot_config}")
    if bot_config is None:
        logging.critical(
            "something very weird happened and i couldn't load config, exiting!"
        )
        sys.exit(2)

    api_key = os.getenv("UNREASONABLE_LLAMA_DISCORD_API_KEY")
    if api_key is None:
        logging.critical(
            "Couldn't load API key from environmental variable UNREASONABLE_LLAMA_DISCORD_API_KEY, exiting!"
        )
        sys.exit(3)

    client = SteelLlamaDiscordClient(bot_config)
    client.run(api_key)


if __name__ == "__main__":
    args = parse_script_arguments()
    if args.log_level != "none":
        log_level_mapping = logging.getLevelNamesMapping()
        log_level = log_level_mapping[args.log_level.upper()]
        logging.basicConfig(level=log_level)
    main(args)
