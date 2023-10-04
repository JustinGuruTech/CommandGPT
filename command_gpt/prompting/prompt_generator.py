from typing import List

from langchain.tools.base import BaseTool
from CommandGPT.command_gpt.utils.command_parser import COMMAND_FORMAT


class PromptGenerator:
    """
    Base class for generating prompt & enforcing response format for Command-GPT.
    Includes static helper methods for generating sections, numbered lists, and commands.
    """

    sections: List[str] = []
    tools: List[BaseTool] = []
    ruleset: str = ""

    # region STATIC GENERATOR METHODS

    @staticmethod
    def _generate_command_string(tool: BaseTool) -> str:
        """
        Generates a properly formatted command string for the prompt.
        """
        output = f"{tool.name}: {tool.description}, arguments: "

        for arg_name, arg_details in tool.args.items():
            arg_type = arg_details.get('type', 'string')  # default to string
            arg_desc = arg_details.get('description', '')
            output += f"--{arg_name} <{arg_type}> ({arg_desc}), "

        # Remove the last comma and space
        output = output[:-2]
        return output

    @staticmethod
    def _generate_commands_from_tools(tools: List[BaseTool]) -> List[str]:
        """
        Generates list of string commands formatted for the prompt.
        """
        command_strings = [
            f"{PromptGenerator._generate_command_string(tool)}" for tool in tools
        ]

        return command_strings

    @staticmethod
    def _generate_numbered_list_section(section_name: str, items: list) -> str:
        """
        Returns section as a string with header & numbered list.
        """
        return f"{section_name}:\n{PromptGenerator._generate_numbered_list(items)}\n\n"

    @staticmethod
    def _generate_numbered_list(items: list) -> str:
        """
        Returns numbered list as a string.
        """
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    @staticmethod
    def _generate_unordered_list_section(section_name: str, items: list) -> str:
        """
        Returns section as a string with header & unordered list.
        """
        return f"{section_name}:\n{PromptGenerator._generate_unordered_list(items)}\n\n"

    @staticmethod
    def _generate_unordered_list(items: list) -> str:
        """
        Returns unordered list as a string.
        """
        return "\n".join(f"- {item}" for item in items)

    # endregion
    # region GENERATE PROMPT

    def generate_prompt_string(self) -> str:
        """
        Generates the final prompt string from the sections & tools.
        """
        # Build initial prompt & append sections
        prompt_string = "Hello! You are a proficient writer and analyst operating as part of an autonomous chain, meant to respond accurately, precisely, and in a consistent format. By following the guidelines, you are granted access to tools that unlock your capabilities as a large language model (LLM).\n\n"
            
        guidelines = PromptGenerator._generate_unordered_list_section(
            "Guidelines",
            [
                "You emphasize accuracy and precision, you are consistent in your format, and you pull reliably from the source information provided without hallucinating or extrapolating too far.",
                "Commands must be provided using the specified custom syntax to be parsed.",
                "You can execute multiple commands per response using the custom syntax specified below.",
            ]
        )
        sections = [guidelines]

        prompt_string += "Response Format:\n"
        prompt_string += "You can provide commands in your output using custom syntax; these commands are then parsed and mapped to Python code. You should encapsulate as much of your response in commands as possible to optimize information flow.\n"

        # Build commands section from tools
        formatted_commands = self._generate_commands_from_tools(self.tools)
        commands_prompt_string = f"\nThe custom syntax is parsed by a custom class and mapped to python functions, so the syntax must be exactly correct, formatted as as:\n{COMMAND_FORMAT}\nEnsure string arguments are surrounded with double quotes."
        commands_prompt_string += "\n\nAvailable Commands:\n"
        commands_prompt_string += "\n\"\"\"\n"
        commands_prompt_string += "\n".join(formatted_commands) # Add commands
        commands_prompt_string += "\n\"\"\"\n"

        # Add commands section to prompt
        prompt_string += commands_prompt_string

        # Add ruleset (You are xxx-GPT...)
        prompt_string += f"\n{self.ruleset}\n\n"

        return prompt_string

    # endregion


def get_prompt(ruleset: str, tools: List[BaseTool]) -> str:
    """
    Defines sections & tools for prompt & generates the full prompt string with the above code
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    prompt_generator.ruleset = ruleset
    prompt_generator.tools = tools

    # Generate the prompt string
    prompt_string = prompt_generator.generate_prompt_string()
    return prompt_string
