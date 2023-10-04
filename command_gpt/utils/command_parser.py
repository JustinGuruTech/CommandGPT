from abc import abstractmethod
import re
import shlex
from typing import Dict, NamedTuple

from langchain.schema import BaseOutputParser

# Command response format
COMMAND_LINE_START = "<cmd>"
COMMAND_LINE_END = "</cmd>"
COMMAND_FORMAT = f"{COMMAND_LINE_START}command_name --arg_string \"string_value\" --arg_int 2{COMMAND_LINE_END}"


class GPTCommand(NamedTuple):
    name: str
    args: Dict


class BaseCommandGPTOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> GPTCommand:
        """Return GPTCommand"""


def preprocess_json_input(input_str: str) -> str:
    """
    Replace single backslashes with double backslashes,
    while leaving already escaped ones intact
    """
    corrected_str = re.sub(
        r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", input_str
    )
    return corrected_str


class CommandGPTOutputParser(BaseCommandGPTOutputParser):
    """
    Custom Parser for CommandGPT that extracts the command string from the response and maps it to a GPTCommand.
    """

    def parse(self, text: str) -> list[GPTCommand]:
        commands = []
        try:
            start_indices = [m.start()
                             for m in re.finditer(COMMAND_LINE_START, text)]
            end_indices = [m.start()
                           for m in re.finditer(COMMAND_LINE_END, text)]

            if len(start_indices) != len(end_indices):
                raise ValueError("Mismatched command start and end tags")

            for start, end in zip(start_indices, end_indices):
                cmd_str = text[start + len(COMMAND_LINE_START):end].strip()
                if cmd_str.startswith('\n'):
                    cmd_str = cmd_str[1:]

                cmd_str_splitted = shlex.split(cmd_str)

                if len(cmd_str_splitted) < 1:
                    raise ValueError(
                        "Command line format error: Missing command name")

                command_name = cmd_str_splitted.pop(0)
                command_args = {}

                while cmd_str_splitted:
                    arg = cmd_str_splitted.pop(0).lstrip('-')
                    if cmd_str_splitted and not cmd_str_splitted[0].startswith('--'):
                        value = cmd_str_splitted.pop(0)
                    else:
                        value = "true"

                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isnumeric():
                        value = int(value)

                    command_args[arg] = value

                commands.append(GPTCommand(
                    name=command_name, args=command_args))

        except Exception as e:
            commands.append(GPTCommand(name="ERROR", args={"error": str(e)}))

        return commands
