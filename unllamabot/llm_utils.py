"""LLM-related utility functions and classes"""

from bot_database import ChatRole, Message
from jinja2 import Template
from llama_backend import LlamaBackend


class LLMUtils:
    def __init__(self, backend: LlamaBackend) -> None:
        self.backend = backend

    def format_messages_into_chat(self, messages: list[Message]) -> str:
        model_props = self.backend.model_props()
        chat_template = Template(model_props.chat_template)
        template_args = {
            "add_generation_prompt": messages[-1].role == ChatRole.USER,
            "messages": [{"role": str(message.role), "content": message.message} for message in messages],
        }
        return chat_template.render(template_args)
