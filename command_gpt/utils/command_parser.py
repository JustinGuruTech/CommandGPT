import json
from abc import abstractmethod
import re
from typing import Dict, NamedTuple

from langchain.schema import BaseOutputParser

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
    Custom Parser for CommandGPT that extracts the command string from the response
    Assumes response will contain a command line represented at the end
    With ```cmd> {"command": {...}}```
    """
    def parse(self, text: str) -> GPTCommand:
        """
        Extract the command string from the response & return GPTCommand
        """
        start_marker = "```cmd>"
        end_marker = "```"

        start_index = text.find(start_marker)
        end_index = text.find(end_marker, start_index + len(start_marker))

        if start_index != -1 and end_index != -1:
            cmd_str = text[start_index + len(start_marker):end_index].strip()
        else:
            return GPTCommand(
                name="ERROR",
                args={"error": "No command found in the response"},
            )

        try:
            parsed = json.loads(cmd_str, strict=False)
        except json.JSONDecodeError:
            preprocessed_text = preprocess_json_input(cmd_str)
            try:
                parsed = json.loads(preprocessed_text, strict=False)
            except Exception:
                return GPTCommand(
                    name="ERROR",
                    args={"error": f"Could not parse invalid json: {cmd_str}"},
                )
        try:
            command = parsed["command"]
            return GPTCommand(
                name=command["name"],
                args=command["args"],
            )
        except (KeyError, TypeError):
            # If the command is null or incomplete, return an erroneous tool
            return GPTCommand(
                name="ERROR", args={"error": f"Incomplete command args: {parsed}"}
            )
