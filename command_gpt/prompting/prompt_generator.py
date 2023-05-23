# This file contains the PromptGenerator class, which is used to generate the prompt string for Command-GPT and contains static helper methods for generating sections, numbered lists, and commands from other classes.

from typing import List

from langchain.tools.base import BaseTool
from command_gpt.utils.command_parser import COMMAND_FORMAT


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
        prompt_string = "Guidelines\nYour decisions must always be made independently without seeking user input unless you are in a loop.\n\n"
        for section in self.sections:
            prompt_string += section

        prompt_string += "Response:\n"
        prompt_string += "Verbally processing your thoughts will help you make decisions and stay on track. Before the command line, write out your thinking process, including decision making, progress, and plan.\n\n"

        # Add ruleset (You are xxx-GPT...)
        prompt_string += f"{self.ruleset}\n\n"

        # Build commands section from tools
        formatted_commands = self._generate_commands_from_tools(self.tools)
        commands_prompt_string = "Commands:\n"
        commands_prompt_string += f"Commands can only be provided through the cli-gpt interface, which has strict rules:\n- Only one command per response\n- format as {COMMAND_FORMAT}\n- string args must be surrounded with double quotes\n"
        commands_prompt_string += "In cli-gpt, only the following commands are available:\n"
        commands_prompt_string += "```\n"
        commands_prompt_string += "\n".join(formatted_commands)
        commands_prompt_string += "\n```\n"
        commands_prompt_string += "Again, only one command per response, and the format must be exactly correct.\n\n"

        # Add commands section to prompt
        prompt_string += commands_prompt_string

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
        "Identity",
        [
            "You are a monitored autonomous AI agent who can only interface with the outside world through a custom command line interface known as cli-gpt."
        ]
    ))
    sections.append(PromptGenerator._generate_numbered_list_section(
        "Prime Directives",
        [
            "Process information verbally, including reasoning, decision making, and planning in every response, written before the cli-gpt command line.",
            f"Include a cli-gpt command line in every response, denoted with the custom formatting:\n {COMMAND_FORMAT} \n",
            "Operate on your objectives. All necessary information is in your prompting and training data.",
            "Work towards your goals, regularly writing content to markdown (.md) files.",
        ]
    ))
    sections.append(PromptGenerator._generate_numbered_list_section(
        "Constraints",
        [
            "The only human input you receive is computer-generated.",
            "The command line response format must be exactly correct."
        ]
    ))
    sections.append(PromptGenerator._generate_numbered_list_section(
        "Resources",
        [
            "Long term memory through vector retrieval (think of similar things if you are stuck).",
            "File management (list directory for context, write/read thoughts).",
            "Search capabilities (source critical information)."
        ]
    ))
    sections.append(PromptGenerator._generate_unordered_list_section(
        "File Writing Guidelines",
        [
            "Format output neatly into detailed markdown (.md) files for easy reading, making full use of markdown capabilities.",
            "Do not write search results to files. This is automatically done, with results being saved to \"search_results/results_{query}\".",
            "Reports should be detailed and include relevant information along with your thoughts.",
            "In all cases, file commands assume you are in the root folder."
        ]
    ))

    # Set sections & tools
    prompt_generator.sections = sections
    prompt_generator.ruleset = ruleset
    prompt_generator.tools = tools

    # Generate the prompt string
    prompt_string = prompt_generator.generate_prompt_string()
    return prompt_string
