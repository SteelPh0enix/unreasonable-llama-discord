"""Utilities for interacting with Discord"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import discord
from bot_config import BotConfig
from bot_core import UnreasonableLlamaBot
from llama_backend import split_message


def current_time_ms() -> int:
    return round(time.time() * 1000)


def user_is_admin(user: discord.User | discord.Member, config: BotConfig) -> bool:
    return user.id in config.admins_id


def requires_admin_permission(func: Callable) -> Callable:  # type: ignore
    async def wrapper(self, message: discord.Message, *args, **kwargs):  # type: ignore
        if user_is_admin(message.author, self.config):
            await self.bot_reply(message, "You do not have permission to use this command.")
            return
        else:
            await func(self, message, *args, **kwargs)

    return wrapper


class UnreasonableLlamaDiscordClient(discord.Client):
    def __init__(self, bot: UnreasonableLlamaBot) -> None:
        self.bot = bot
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

    async def bot_reply(self, message_to_reply_to: discord.Message, reply_content: str) -> discord.Message:
        """Replies to a message and adds bot-related stuff, like emojis, to reply"""
        reply = await message_to_reply_to.reply(content=reply_content)
        await reply.add_reaction(self.bot.config.message_removal_reaction)
        return reply

    async def chained_reply(self, message_to_reply_to: discord.Message, reply_content: str) -> None:
        """Use this method to reply with more than `self.bot.config.message_length_limit` characters"""
        first, second = split_message(reply_content, self.bot.config.message_length_limit)
        message_to_reply_to = await self.bot_reply(message_to_reply_to, first)
        if second is not None:
            await self.chained_reply(message_to_reply_to, second)

    async def update_bot_presence(self) -> None:
        model_props = self.bot.backend.model_props()
        model_name = Path(model_props.default_generation_settings.model).name
        model_context_length = model_props.default_generation_settings.n_ctx
        logging.info(f"Loaded model: {model_name}")

        presence_string = (
            f"Chat with me using `{self.bot.config.bot_prefix}{self.bot.config.commands['inference'].command}`! "
            + f"Currently using {model_name} with context of {model_context_length} tokens per user."
        )
        logging.info(f"Bot presence: {presence_string}")
        await self.change_presence(activity=discord.CustomActivity(presence_string))
        logging.info("Bot is ready!")

    async def on_ready(self) -> None:
        if not self.bot.backend.is_alive():
            raise RuntimeError("Backend is not running, or configured IP is invalid!")
        await self.update_bot_presence()

    async def process_inference_command(self, message: discord.Message, prompt: str | None) -> None:
        if prompt is None:
            prefixed_inference_command = f"{self.bot.config.bot_prefix}{self.bot.config.commands['inference']}"
            prefixed_help_command = f"{self.bot.config.bot_prefix}{self.bot.config.commands['help']}"
            await self.bot_reply(
                message,
                f"*Usage: `{prefixed_inference_command} [message]`, for example `{prefixed_inference_command} what's the highest mountain on earth?`*\n"
                f"*Use `{prefixed_help_command}` for details about the bot commands.*",
            )
            return

        # This will be a placeholder, LLM windup can take a while.
        reply_message = await self.bot_reply(message, "*Generating response, please wait...*")
        last_update_time = current_time_ms()

        async for chunk in self.bot.process_message(prompt, message.author.id):
            if chunk.new_message:
                reply_message = await self.bot_reply(reply_message, chunk.message)
            else:
                if len(chunk.message.strip()) > 0 and (
                    chunk.end_of_message
                    or current_time_ms() - last_update_time >= self.bot.config.message_edit_cooldown
                ):
                    reply_message = await reply_message.edit(content=chunk.message)
                    last_update_time = current_time_ms()

            if chunk.end_of_response:
                reply_message = await reply_message.edit(content=chunk.message)

    async def process_help_command(self, message: discord.Message, subject: str | None = None) -> None:
        help_content = f"*No help available for selected subject. Try {self.bot.config.bot_prefix}{self.bot.config.commands['help']} for list of subjects and generic help.*"

        match subject:
            case None:
                help_content = f"""# This is [UnreasonableLlama](https://pypi.org/project/unreasonable-llama/)-based Discord bot.
It allows you to converse with an LLM hosted via llama.cpp.
The bot remembers your conversations and allows you to configure the LLM in some degree.

[This bot is open-source](https://github.com/SteelPh0enix/unreasonable-llama-discord).
## Available commands:
    * `{self.bot.config.bot_prefix}{self.bot.config.commands["inference"].command} [message]` - chat with the LLM
    * `{self.bot.config.bot_prefix}{self.bot.config.commands["help"].command} [subject (optional)]` - show help
    * `{self.bot.config.bot_prefix}{self.bot.config.commands["reset-conversation"].command}` - clear your conversation history and start a new one
    * `{self.bot.config.bot_prefix}{self.bot.config.commands["stats"].command}` - show some stats of your conversation
    * `{self.bot.config.bot_prefix}{self.bot.config.commands["get-param"].command} [param (optional)]` - show your LLM parameters/configuration

    * `{self.bot.config.bot_prefix}{self.bot.config.commands["set-param"].command} [param] [new value]` - set your LLM parameters/configuration
    * `{self.bot.config.bot_prefix}{self.bot.config.commands["reset-param"].command} [param]` - Reset your LLM parameter to default value

When chatting directly with the bot, the messages are automatically passed as argument for `{self.bot.config.bot_prefix}{self.bot.config.commands["inference"].command}` command, if they don't match any other command.
## Additional help subjects:
    * `model` - show model details
    * `params` - show available LLM parameters and their description
    * `admin` - show admin commands"""
            case "model":
                props = self.bot.backend.model_props()
                model_info = props.default_generation_settings
                help_content = f"""# Currently loaded model: {Path(model_info.model).name}
**Context length**: {model_info.n_ctx} tokens
**Default system prompt**: `{self.bot.config.default_system_prompt}`
**Samplers**: {model_info.samplers}
## Default LLM parameters
**Top-K**: {model_info.top_k}
**Typical-P**: {model_info.typical_p:.02f}
**Top-P**: {model_info.top_p:.02f}
**Min-P**: {model_info.min_p:.02f}
**Temperature**: {model_info.temperature:.02f}
**Mirostat type**: {model_info.mirostat}
**Mirostat learning rate (Eta)**: {model_info.mirostat_eta:.02f}
**Mirostat target entropy (Tau)**: {model_info.mirostat_tau:.02f}"""
            case "admin":
                if user_is_admin(message.author, self.bot.config):
                    help_content = f"""# Admin commands
* `{self.bot.config.bot_prefix}{self.bot.config.commands["refresh"].command}` - refresh llama.cpp props"""
                else:
                    help_content = "You lack permissions to check for this subject."
            case "params":
                help_content = f"""# (incomplete) List of LLM parameters (configuration is NOT available yet!)
* `system-prompt`: System prompt for the LLM. Defines the behaviour of LLM and the character of it's responses.
* `temperature`: Temperature controls the probability distribution of tokens selected by LLM. Lower temperature reduces randomness of tokens selected by LLM (tokens with high probability have higher change of being selected, and vice-versa), while higher temperature increases it by making the chance of selecting a token with high probability lower (and vice-versa). In other words, lower temperature implies more deterministic output, and higher temperature implies more diverse output. **Recommended range: (0, 2]**.
* `dynatemp_range` (dynamic temperature range): llama.cpp implements [entropy-based dynamic temperature sampling](https://arxiv.org/pdf/2403.14541v1). This parameter, if non-zero, defines the range of temperature range used during token prediction as `[temperature - dynatemp_range, temperature + dynatemp_range]`, with lower range capped at 0.
* `dynatemp_exponent`: (dynamic temperature exponent): This parameter is the exponent of normalized entropy used during dynamic temperature calculation. See paper linked in `dynatemp_range`, or `llama-sampling.cpp` file from `llama.cpp` repo for more details.
* `top_k`: Top-K parameter limits the number of tokens considered by LLM during prediction step to specified value. Lower values will produce more deterministic and focused output, higher - more diverse and creative. More details can be found in [this paper](https://arxiv.org/pdf/1904.09751). **Recommended range: [10, 100]**
* `top_p` (nucleus sampling): Top-P parameter adjusts the number of tokens based on their cumulative probability. High values of top-p will allow the LLM to use more tokens during generation - leading to more diverse text, while lower values will limit the amount of used tokens, leading to more focused output. **Recommended range: (0, 1)**
* `min_p`: Min-P parameter defines the minimum probability of a token to be considered during prediction. More details can be found in [this paper](https://arxiv.org/pdf/2407.01082). **Recommended range: (0, 1)**
* `xtc_threshold` ([Exclude Top Choices](https://github.com/oobabooga/text-generation-webui/pull/6335)): "If there are multiple tokens with predicted probability at least `xtc_threshold`..."
* `xtc_probability` ([Exclude Top Choices](https://github.com/oobabooga/text-generation-webui/pull/6335)): "...remove all except the least probable one from sampling, with probability `xtc_probabililty`"
* `n_predict`: Defines the amount of tokens to predict. Default value will allow the LLM to define the end of sentence.
* `n_keep`: Defines the amount of tokens to retains from original prompt, when LLM runs out of context. -1 will force llama.cpp to retain all tokens from initial prompt.
* `tfs_z` ([tail-free sampling](https://www.trentonbricken.com/Tail-Free-Sampling/)): TFS removes the tokens with less-than-desired *second derivative* of token's probability from token pool used during prediction. It's similar to top-p sampling, except top-p uses the probabilities themselves instead of derivatives. More details can be found in linked article. **Recommended range: (0, 1], where 1 == TFS disabled**
* `typical_p` ([locally typical sampling](https://arxiv.org/pdf/2202.00666)): Locally typical sampling can increase diversity of the text without major coherence degradation by choosing tokens that are typical or expected based on the context. For more details, see linked paper. **Recommended values: (0, 1]**, where 1 == disabled.
* `repeat_penalty`: Penalty for repeating the tokens in generated text. Higher values will penalize the repeated tokens more. **Recommended values: (0, 5)**
* `repeat_last_n`: Amount of last tokens to penalize for repetition. Setting this to 0 disabled penalization, and -1 penalizes the whole context.
* `penalize_ln`: Enable/disable penalization of newline tokens. Accepts values 1, 0, `true`, `false`, `yes` and `no` (case-insensitive).
* `presence_penalty`: Penalty for re-using the tokens that are already present in generated text. Higher values lead to more creative text. **Recommended values: [0, 2)**, where 0 == presence penalty disabled
* `frequency_penalty`: Penalty applied for re-using the tokens that are already present in generated text, based on the frequency of their appearance. Similar to presence penalty. **Recommended values: [0, 2)**, where 0 == frequency penalty disabled.
* `mirostat` ([Mirostat sampling](https://arxiv.org/pdf/2007.14966)): Mirostat is an algorithm that actively maintains the quality of generated text within a desired range during text generation. It aims to strike a balance between coherence and diversity, avoiding low-quality output caused by excessive repetition (boredom traps) or incoherence (confusion traps) [(source)[https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#mirostat-sampling]]. **Valid values: 0 (disabled), 1 (Mirostat), 2 (Mirostat 2.0)**. **ENABLING MIROSTAT DISABLES OTHER SAMPLERS!!!**
* `mirostat_tau`: Mirostat target entropy, desired perplexity for generated text. Lower values lead to more coherent and focused output, while higher will generate more creative one. **Recommended values: (0, 10)**
* `mirostat_eta`: Mirostat learning rate. Influences how fast algorithm responds to feedback. **Recommended values: (0, 1)**
* `seed`: 32-bit seed used for RNG.
* `samplers`: List of samplers to use, in order of usage. List should contain names of samplers separated by commas, for example `top_k, tfs_z, typ_p, top_p, min_p, temperature`. Whitespace is ignored. **Valid samplers:**
    * `top_k`
    * `tfs_z`
    * `typ_p`
    * `top_p`
    * `min_p`
    * `temperature`
## Checking and modifying LLM parameters
All the parameters listed above are stored on per-user basis, therefore you are able to modify them to your liking.
To check current value of the parameter, use `{self.bot.config.bot_prefix}{self.bot.config.commands["get-param"].command} [parameter-name]` command, for example, `{self.bot.config.bot_prefix}{self.bot.config.commands["get-param"].command} system-prompt`.
You can also use `{self.bot.config.bot_prefix}{self.bot.config.commands["get-param"].command}` to list the values of all parameters.
To change the value of the parameter, use `{self.bot.config.bot_prefix}{self.bot.config.commands["set-param"].command} [parameter-name] [new value]`, for example `{self.bot.config.bot_prefix}{self.bot.config.commands["set-param"].command} system-prompt This is my new system prompt!`.
The change is immediate, resetting the conversation is not required.
You can use `{self.bot.config.bot_prefix}{self.bot.config.commands["reset-param"].command}` to reset the parameter it's default value.
"""

        await self.chained_reply(message, help_content)

    async def process_reset_conversation_command(self, message: discord.Message) -> None:
        self.bot.db.clear_user_messages(message.author.id)
        await self.bot_reply(message, "Message history cleared!")

    async def process_stats_command(self, message: discord.Message) -> None:
        stats = self.bot.get_user_stats(message.author.id)
        await self.chained_reply(
            message,
            f"""Messages in chat history (including system prompt): {stats.messages_in_chat_history}
Current prompt length (tokens): {stats.chat_length_tokens}/{stats.context_length} ({stats.context_percent_used:.2f}% of available context used)
Current prompt length (characters): {stats.chat_length_chars}""",
        )

    @requires_admin_permission
    async def process_refresh_command(self, message: discord.Message) -> None:
        await self.update_bot_presence()
        await self.bot_reply(message, "Bot's metadata refreshed!")

    async def process_get_param(self, message: discord.Message, param: str | None = None) -> None:
        user = self.bot.db.get_or_create_user(message.author.id)
        response_content = ""
        match param:
            case "system-prompt":
                response_content = f"Your system prompt is `{user.system_prompt}`"
            case None:
                response_content = f"* **System prompt**: `{user.system_prompt}`"
            case _:
                response_content = f"Unknown parameter: {param}"

        await self.chained_reply(message, response_content)

    async def process_set_param(self, message: discord.Message, params: str | None) -> None:
        if params is None:
            await self.bot_reply(message, "Missing parameter name and new value!")
            return

        try:
            param_name, new_param_value = params.split(sep=" ", maxsplit=1)
        except ValueError:
            await self.bot_reply(message, "Missing value of the parameter!")
            return

        user = self.bot.db.get_or_create_user(message.author.id)
        match param_name:
            case "system-prompt":
                self.bot.db.change_user_system_prompt(user.id, new_param_value)
                await self.chained_reply(
                    message,
                    f"Updated system prompt!\nOld: ```\n{user.system_prompt}\n```\nNew: ```\n{new_param_value}```",
                )
            case _:
                await self.bot_reply(message, f"Unknown parameter: {param_name}")

    async def process_reset_param(self, message: discord.Message, param: str | None) -> None:
        if param is None:
            await self.bot_reply(message, "Missing parameter name!")
            return

        match param:
            case "system-prompt":
                await self.process_set_param(message, f"system-prompt {self.bot.config.default_system_prompt}")
            case _:
                await self.bot_reply(message, f"Unknown parameter: {param}")

    async def on_message(self, message: discord.Message) -> None:
        # ignore your own messages
        if message.author == self.user:
            return

        # ignore messages without prefix, unless in DMs
        msg_starts_with_prefix = message.content.startswith(self.bot.config.bot_prefix)
        msg_is_dm = isinstance(message.channel, discord.DMChannel)
        if not msg_starts_with_prefix and not msg_is_dm:
            return

        command_parts = message.content.lstrip(self.bot.config.bot_prefix).split(" ", 1)
        command_name = command_parts[0]
        arguments = command_parts[1] if len(command_parts) > 1 else None

        logging.info(
            f"<UID:{message.author.id}|UN:{message.author.global_name}> Command detected: {command_name}, arguments: {arguments}"
        )

        if command_name == self.bot.config.commands["inference"].command:
            await self.process_inference_command(message, arguments)
        elif command_name == self.bot.config.commands["help"].command:
            await self.process_help_command(message, arguments)
        elif command_name == self.bot.config.commands["reset-conversation"].command:
            await self.process_reset_conversation_command(message)
        elif command_name == self.bot.config.commands["stats"].command:
            await self.process_stats_command(message)
        elif command_name == self.bot.config.commands["refresh"].command:
            await self.process_refresh_command(message)
        elif command_name == self.bot.config.commands["get-param"].command:
            await self.process_get_param(message, arguments)
        elif command_name == self.bot.config.commands["set-param"].command:
            await self.process_set_param(message, arguments)
        elif command_name == self.bot.config.commands["reset-param"].command:
            await self.process_reset_param(message, arguments)
        else:
            if msg_is_dm:
                await self.process_inference_command(message, message.content)
            else:
                await self.bot_reply(message, f"Unknown command: {command_name}")

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

        if str(event.emoji) == self.bot.config.message_removal_reaction:
            message_channel = await self.fetch_channel(event.channel_id)
            if isinstance(
                message_channel,
                discord.TextChannel | discord.DMChannel | discord.GroupChannel | discord.Thread,
            ):
                message_to_delete = await message_channel.fetch_message(event.message_id)
                logging.info(f"Removing message {message_to_delete.id}")
                await message_to_delete.delete()
            else:
                logging.warning(
                    f"Message removal emoji received from channel {message_channel} - cannot fetch and delete the target message!"
                )
