from __future__ import annotations
from typing import List, Optional
from pydantic import ValidationError

from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    Document,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever

from CommandGPT.command_gpt.utils.command_parser import GPTCommand, CommandGPTOutputParser, COMMAND_FORMAT
from CommandGPT.command_gpt.utils.console_logger import ConsoleLogger
from CommandGPT.command_gpt.prompting.prompt import CommandGPTPrompt


class CommandGPT:
    """Agent class driving CommandGPT loop"""

    def __init__(
        self,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: CommandGPTOutputParser,
        tools: List[BaseTool],
    ):
        self.memory = memory
        self.full_message_history: List[BaseMessage] = []
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools

    @classmethod
    def from_ruleset_and_tools(
        cls,
        ruleset: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        output_parser: Optional[CommandGPTOutputParser] = None,
    ) -> CommandGPT:
        prompt = CommandGPTPrompt(
            ruleset=ruleset,
            tools=tools,
            input_variables=["memory", "messages", "user_input"],
            token_counter=llm.get_num_tokens,
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        return cls(
            memory,
            chain,
            output_parser or CommandGPTOutputParser(),
            tools,
        )

    def run(self) -> str:
        """
        Kicks off interaction loop with AI
        """

        # Interaction Loop
        loop_count = 0
        while True:
            loop_count += 1
            messages = self.full_message_history

            # todo: pass contextual information to AI
            system_message = f"Current loop count: {loop_count}\nUse commands to achieve the defined goals."
            self.full_message_history.append(
                SystemMessage(content=system_message))

            user_input = input(
                "Any additional commands? (Press Enter to continue): ")
            if user_input.strip():  # If not empty
                self.full_message_history.append(
                    HumanMessage(content=user_input))

            # Set response color for console logger
            ConsoleLogger.set_response_stream_color()
            # Send message to AI, get response
            assistant_reply = self.chain.run(
                messages=messages,
                memory=self.memory,
                user_input=system_message,
            )

            # Update message history
            self.full_message_history.append(
                HumanMessage(content=system_message))
            # self.full_message_history.append(SystemMessage(content=system_message))
            self.full_message_history.append(
                AIMessage(content=assistant_reply))

            # Parse commands and execute
            tools = {t.name: t for t in self.tools}
            # Now returns a list of commands
            actions = self.output_parser.parse(assistant_reply)

            for action in actions:
                command_result = self.try_execute_command(tools, action)

                memory_to_add = (
                    f"Assistant Reply: {assistant_reply} " f"\nResult: {command_result} "
                )

                self.memory.add_documents(
                    [Document(page_content=memory_to_add)])
                self.full_message_history.append(
                    SystemMessage(content=command_result))

    def try_execute_command(self, tools_available: List[BaseTool], command: GPTCommand):
        """
        Executes a command if available in tools, otherwise returns an error message
        """
        if command.name == "finish":
            return command.args["tool_input"]
        if command.name in tools_available:
            tool = tools_available[command.name]
            try:
                observation = tool.run(command.args)
            except ValidationError as e:
                observation = (
                    f"Validation Error in args: {str(e)}, args: {command.args}"
                )
            except Exception as e:
                observation = (
                    f"Error: {str(e)}, {type(e).__name__}, args: {command.args}"
                )
            result = f"Command {tool.name} returned: {observation}"
        elif command.name == "ERROR":
            ConsoleLogger.log_error("Command not parsed")
            result = f"Improperly formatted command line. Proper formatting:\n\n {COMMAND_FORMAT}"
        else:
            result = (
                f"Unknown command '{command.name}'. Please refer to the Commands list for available commands and only respond in the specified command line format."
            )
        return result
