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
        prompt_string = "Hello! You are an AI designed as part of a semi-autonomous system, meant to respond accurately, precisely, and in a consistent format. By following the guidelines, you are able to use tools that unlock your capabilities as a large language model (LLM) and help you interact with the world.\n\n"
        for section in self.sections:
            prompt_string += section

        prompt_string += "Response Format:\n"
        prompt_string += "You can execute multiple commands per response by using the custom syntax established here.\n"

        # Build commands section from tools
        formatted_commands = self._generate_commands_from_tools(self.tools)
        commands_prompt_string = f"The custom syntax is parsed by a custom class and mapped to python functions, so the syntax must be exactly correct. Format commands as {COMMAND_FORMAT} and ensure string arguments are surrounded with double quotes.\n"
        commands_prompt_string += "The following commands are available to you:\n"
        commands_prompt_string += "```\n"
        commands_prompt_string += "\n".join(formatted_commands)
        commands_prompt_string += "\n```\n"
        commands_prompt_string += "Many commands can be parsed per response as long as the format is exactly correct.\n"

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

    # Build sections
    sections = []
    sections.append(PromptGenerator._generate_unordered_list_section(
        "Guidelines",
        [
            "The command line response format must be exactly correct.",
            "You can execute multiple commands per response by using the required syntax, specified below.",
            "You emphasize accuracy and precision, you are consistent in your format, and you pull reliably from the information established in the guidelines.",
        ]
    ))

    # Set sections & tools
    prompt_generator.sections = sections
    prompt_generator.ruleset = ruleset
    prompt_generator.tools = tools

    # Generate the prompt string
    prompt_string = prompt_generator.generate_prompt_string()
    return prompt_string
