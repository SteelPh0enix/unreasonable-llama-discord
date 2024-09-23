"""LLM-related utility functions and classes"""

from transformers import AutoTokenizer
from bot_database import ChatRole, Message


class LLMUtils:
    def __init__(self, model_path_or_url: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_url)

    def format_messages_into_chat(self, messages: list[Message]) -> str:
        conversation = [{"role": msg.role, "content": msg.message} for msg in messages]
        last_message = messages[-1]

        add_generation_prompt = last_message.role == ChatRole.USER
        continue_generation = last_message.role == ChatRole.BOT

        return str(
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_generation,
            )
        )
