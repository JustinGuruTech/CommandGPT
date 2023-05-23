# Full prompt with base prompt, time, memory, and historical messages

import time
from typing import Any, Callable, List

from pydantic import BaseModel

from langchain.prompts.chat import (
    BaseChatPromptTemplate,
)
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever

from command_gpt.prompting.prompt_generator import get_prompt
from command_gpt.utils.console_logger import ConsoleLogger


class CommandGPTPrompt(BaseChatPromptTemplate, BaseModel):
    """
    Prompt template for Command-GPT with base prompt, time, memory, and historical messages. Gets full prompt from prompt_generator.py.
    """
    ruleset: str
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196

    # todo: probably move to command_gpt.py for more holistic logging
    # Always log full prompt on first run
    is_first_run = True

    def construct_full_prompt(self) -> str:
        # Construct full prompt
        full_prompt = f"\n\n{get_prompt(self.ruleset, self.tools)}"

        # Log full prompt on first run
        if self.is_first_run:
            ConsoleLogger.log_input(full_prompt)
            self.is_first_run = False
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        base_prompt = SystemMessage(content=self.construct_full_prompt())
        time_prompt = SystemMessage(
            content=f"The current time and date is {time.strftime('%c')}"
        )
        used_tokens = self.token_counter(base_prompt.content) + self.token_counter(
            time_prompt.content
        )

        # Get relevant memory & format into message
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]
        relevant_docs = memory.get_relevant_documents(
            str(previous_messages[-10:]))
        relevant_memory = [d.page_content for d in relevant_docs]
        relevant_memory_tokens = sum(
            [self.token_counter(doc) for doc in relevant_memory]
        )
        while used_tokens + relevant_memory_tokens > 2500:
            relevant_memory = relevant_memory[:-1]
            relevant_memory_tokens = sum(
                [self.token_counter(doc) for doc in relevant_memory]
            )
        content_format = (
            f"This reminds you of these events "
            f"from your past:\n{relevant_memory}\n\n"
        )
        memory_message = SystemMessage(content=content_format)
        used_tokens += self.token_counter(memory_message.content)

        # Append historical messages if there is space
        historical_messages: List[BaseMessage] = []
        for message in previous_messages[-10:][::-1]:
            message_tokens = self.token_counter(message.content)
            if used_tokens + message_tokens > self.send_token_limit - 1000:
                break
            historical_messages = [message] + historical_messages
            used_tokens += message_tokens

        input_message = HumanMessage(content=kwargs["user_input"])
        messages: List[BaseMessage] = [
            base_prompt, time_prompt, memory_message]
        messages += historical_messages
        messages.append(input_message)
        return messages
