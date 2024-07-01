import os

import discord

from unreasonable_llama import (
    UnreasonableLlama,
    LlamaCompletionRequest,
    LlamaSystemPrompt,
)

llama = UnreasonableLlama()

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

BOT_PREFIX = "$llm"
SYSTEM_PROMPT = LlamaSystemPrompt(
    prompt="You are a Discord bot. You are supposed to help the user with whatever he needs, or just talk to him if he wants.",
    anti_prompt="",
    assistant_name="SteelLlama",
)


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith(BOT_PREFIX):
        prompt = message.content.removeprefix(BOT_PREFIX)
        print(f"Requesting completion for prompt: {prompt}")
        llm_response = llama.get_completion(
            LlamaCompletionRequest(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        )

        print(f"Got LLM response: {llm_response}")
        try:
            if len(llm_response.content) > 0:
                await message.channel.send(llm_response.content)
            else:
                await message.channel.send(
                    f"Oops, something went wrong. This is the response i've got: {llm_response}"
                )
        except Exception as e:
            await message.channel.send(
                f"Oops, something went **TERRIBLY** wrong! I've got an exception: {e}"
            )


client.run(os.getenv("UNREASONABLE_LLAMA_DISCORD_API_KEY"))
