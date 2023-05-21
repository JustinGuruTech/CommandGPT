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
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores.base import VectorStoreRetriever

from command_gpt.utils.command_parser import GPTCommand, CommandGPTOutputParser
from command_gpt.utils.console_logger import ConsoleLogger
from command_gpt.prompting.prompt import CommandGPTPrompt
from command_gpt.utils.evaluate import get_filesystem_representation


class CommandGPT:
    """Agent class driving CommandGPT loop"""

    def __init__(
        self,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: CommandGPTOutputParser,
        tools: List[BaseTool],
        feedback_tool: Optional[HumanInputRun] = None,
    ):
        self.memory = memory
        self.full_message_history: List[BaseMessage] = []
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool

    @classmethod
    def from_ruleset_and_tools(
        cls,
        ruleset: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        human_in_the_loop: bool = False,
        output_parser: Optional[CommandGPTOutputParser] = None,
    ) -> CommandGPT:
        prompt = CommandGPTPrompt(
            ruleset=ruleset,
            tools=tools,
            input_variables=["memory", "messages", "user_input"],
            token_counter=llm.get_num_tokens,
        )
        
        human_feedback_tool = HumanInputRun() if human_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=prompt)

        return cls(
            memory,
            chain,
            output_parser or CommandGPTOutputParser(),
            tools,
            feedback_tool=human_feedback_tool,
        )

    def run(self) -> str:
        """
        Kicks off interaction loop with AI
        """

        system_message = (
            "Determine which command to use & respond using the format specified above:"
        )

        # Interaction Loop
        loop_count = 0
        while True:
            # user_input = ConsoleLogger.input("You: ")
            loop_count += 1

            files = get_filesystem_representation(verbose=False)
                
            messages = self.full_message_history
            messages.append(
                SystemMessage(
                    content=f"Loop count: {loop_count}\nFiles: {files}"
                )
            )

            # Set response color for console logger
            ConsoleLogger.set_response_stream_color()
            # Send message to AI, get response
            assistant_reply = self.chain.run(
                messages=messages,
                memory=self.memory,
                user_input=system_message,
            )

            # Update message history
            self.full_message_history.append(HumanMessage(content=system_message))
            # self.full_message_history.append(SystemMessage(content=system_message))
            self.full_message_history.append(AIMessage(content=assistant_reply))

            # Parse command and execute
            tools = {t.name: t for t in self.tools}
            action = self.output_parser.parse(assistant_reply)
            command_result = self.try_execute_command(tools, action)

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {command_result} "
            )
            # if self.feedback_tool is not None:
            #     feedback = f"\n{self.feedback_tool.run('Input: ')}"
            #     if feedback in {"q", "stop"}:
            #         print("EXITING")
            #         return "EXITING"
            #     memory_to_add += feedback

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.full_message_history.append(SystemMessage(content=command_result))

    def try_execute_command(self, tools_available: list[BaseTool], command: GPTCommand):
        """
        Executes a command if available in tools, otherwise returns an error message
        """
        if command.name == "finish":
            return command.args["response"]
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
            result = f"Error: {command.args}. "
        else:
            result = (
                f"Unknown command '{command.name}'. "
                f"Please refer to the 'COMMANDS' list for available "
                f"commands and only respond in the specified JSON format."
            )
        return result