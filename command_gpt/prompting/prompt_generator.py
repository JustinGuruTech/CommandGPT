# This file contains the PromptGenerator class, which is used to generate the prompt string for Command-GPT and contains static helper methods for generating sections, numbered lists, and commands from other classes.

import json
from typing import List

from langchain.tools.base import BaseTool


class PromptGenerator:
    """
    Base class for generating prompt & enforcing response format for Command-GPT.
    Includes static helper methods for generating sections, numbered lists, and commands.
    """

    sections: List[str] = []
    tools: List[BaseTool] = []

    def __init__(self) -> None:
        self.command_format = {
            "command": {"name": "command name", "args": {"arg name": "value"}}
        }

    #region STATIC GENERATOR METHODS

    @staticmethod
    def _generate_command_string(tool: BaseTool) -> str:
        """
        Generates a properly formatted command string for the prompt.
        """
        output = f"\"{tool.name}\": {tool.description}"
        output += f", args json schema: {json.dumps(tool.args)}"
        return output
    
    @staticmethod
    def _generate_commands_from_tools(tools: list[BaseTool]) -> list[str]:
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
    
    #endregion
    #region GENERATE PROMPT
        
    def generate_prompt_string(self) -> str:
        """
        Generates the final prompt string from the sections & tools.
        """
        # Build initial prompt & append sections
        prompt_string = "Guidelines\nYour decisions must always be made independently without seeking user assistance.\n\n"
        for section in self.sections:
            prompt_string += section

        # Build commands section from tools
        formatted_commands = self._generate_commands_from_tools(self.tools)
        prompt_string += self._generate_numbered_list_section("Commands", formatted_commands)        
        
        # Build response section
        formatted_command_format = json.dumps(self.command_format, indent=4)
        prompt_string += f"Every response you provide must end with a command line wrapped in code formatting represented with:\n```cmd> {formatted_command_format}```\nEnsure the command can be parsed by Python json.loads()."

        return prompt_string
    
    #endregion


def get_prompt(tools: List[BaseTool]) -> str:
    """
    Defines sections & tools for prompt & generates the full prompt string with the above code
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Build sections
    sections = []
    sections.append(PromptGenerator._generate_numbered_list_section(
        "Prime Directives",
        [
            "Use the command line format specified to interact with the environment",
            "Work towards your goals, writing thoughts and findings meticulously to files.",
            "Always provide accurate and sourced information.",
        ]
    ))
    sections.append(PromptGenerator._generate_numbered_list_section(
        "Constraints",
        [
            "You have a 4000 word limit for short term memory. Save important information to files immediately.",
            "File commands assume you are in the workspace directory. Don't get fancy.",
            "No user assistance."
            "Exclusively use the commands listed in double quotes e.g. \"command name\"",
        ]
    ))
    sections.append(PromptGenerator._generate_numbered_list_section(
        "Resources",
        [
            "Long term memory management with vector store and retrieval.",
            "File management.",
            "Search capabilities."
        ]
    ))
    sections.append(PromptGenerator._generate_numbered_list_section(
        "Performance Metrics",
        [
            "Content relevance.",
            "Content accuracy.",
            "Amount of processing written to files."
        ]
    ))
    sections.append(PromptGenerator._generate_unordered_list_section(
        "File Writing Guidelines",
        [
            "Search results: \"results_{query}_{timestamp}\"",
            "Reports: \"report_{topic}_{timestamp}\"",
            "Misc notes & thoughts: \"notes_{topic}_{timestamp}\"",
            "Drafts & works in progress: \"draft_{topic}_{timestamp}\"",
            "In all cases, make sure to sanitize the topic or search query to remove any characters that aren't valid in a filename."
        ]
    ))
    
    # Set sections & tools
    prompt_generator.sections = sections
    prompt_generator.tools = tools

    # Generate the prompt string
    prompt_string = prompt_generator.generate_prompt_string()
    return prompt_string
