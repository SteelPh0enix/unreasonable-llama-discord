"""Unreasonable-llama bot's core logic.
Plug a front-end to make it useful."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from collections.abc import AsyncGenerator

from bot_config import BotConfig
from bot_database import BotDatabase, ChatRole
from llama_backend import LlamaBackend, LlamaResponseChunk
from llm_utils import LLMUtils


@dataclass
class UserBotStats:
    messages_in_chat_history: int
    chat_length_chars: int
    chat_length_tokens: int
    context_length: int
    context_percent_used: float


class UnreasonableLlamaBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.db = BotDatabase(config.chat_database_path, config.default_system_prompt)
        self.backend = LlamaBackend(
            config.llama_host,
            config.llama_port,
            config.llama_request_timeout,
        )
        self.llm_utils = LLMUtils(self.backend)

    async def process_message(self, message: str, user_id: int) -> AsyncGenerator[LlamaResponseChunk]:
        user = self.db.get_or_create_user(user_id)
        if not self.db.user_has_messages(user_id):
            self.db.add_message(user_id, ChatRole.SYSTEM, user.system_prompt)
        self.db.add_message(user_id, ChatRole.USER, message)

        user_messages = self.db.get_user_messages(user_id)
        llm_prompt = self.llm_utils.format_messages_into_chat(user_messages)
        full_response = None

        logging.debug(f"Processing message from user {user_id}...")
        logging.debug(f"LLM prompt: {llm_prompt}")

        async for chunk in self.backend.get_buffered_llm_response(llm_prompt, self.config.message_length_limit):
            yield chunk
            if chunk.end_of_response:
                full_response = chunk.response

        logging.debug(f"LLM response: {full_response}")
        self.db.add_message(user_id, ChatRole.BOT, full_response)

    def get_user_stats(self, user_id: int) -> UserBotStats:
        user_messages = self.db.get_user_messages(user_id)
        user_messages_amount = len(user_messages)
        llm_prompt = self.llm_utils.format_messages_into_chat(user_messages)
        tokenized_prompt = self.backend.tokenize(llm_prompt)
        context_length = self.backend.model_props().default_generation_settings.n_ctx
        prompt_length_tokens = len(tokenized_prompt)
        prompt_length_chars = len(llm_prompt)
        context_percent_used = (prompt_length_tokens / context_length) * 100

        return UserBotStats(
            messages_in_chat_history=user_messages_amount,
            chat_length_chars=prompt_length_chars,
            chat_length_tokens=prompt_length_tokens,
            context_length=context_length,
            context_percent_used=context_percent_used,
        )
