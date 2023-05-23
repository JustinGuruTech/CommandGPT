# Defines custom LangChain tools & Bundles tools set up with custom stream callback for printing to console

from pathlib import Path
import re
from typing import Optional

from langchain import GoogleSearchAPIWrapper
from langchain.tools.file_management import (
    WriteFileTool,
)
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)

from config import WORKSPACE_DIR

WORKSPACE_PATH = Path(WORKSPACE_DIR)


class WriteFileToolNewlines(WriteFileTool):
    """
    Extends WriteFileTool to replace any occurrences of (\\x2+)n with \n for clean newlines.
    """

    def _run(
        self,
        file_path: str,
        text: str,
        append: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Replace any occurrences of (\\x2+)n with \n
        text = re.sub(r'\\+n', '\n', text)
        return super()._run(file_path, text, append, run_manager)


class SearchAndWriteTool:
    """
    Wraps the .results() method of a GoogleSearchAPIWrapper in a custom Tool.
    Runs a search query, writes the results to a file, and returns a result message.
    """

    def __init__(self, search: GoogleSearchAPIWrapper):
        self.search = search

    def run(self, query: str) -> str:
        """Run a search query, write the results to a file, and return a result message."""
        # Run the search query
        results = self.search.results(query, num_results=5)
        # Format the results
        results_text = []
        for i, result in enumerate(results, start=1):
            result_text = (
                f"Result {i}:\n"
                f"Title: {self.remove_non_ascii(result['title'])}\n"
                f"Link: {self.remove_non_ascii(result['link'])}\n"
                f"Snippet: {self.remove_non_ascii(result['snippet'])}\n\n"
            )
            results_text.append(result_text)
        results_text = "".join(results_text)
        # Sanitize the query to use it as a filename
        safe_query = self.sanitize_filename(query)
        # Prepare the filename
        file_name = f"search_results/results_{safe_query}.txt"
        file_path = WORKSPACE_PATH / file_name
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Write the results to the file
        with open(file_path, "w") as file:
            file.write(results_text)
        # Return a result message
        return f"Search results:\n\n {results_text}\n\n written to file named: {file_name}"

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """
        Remove any character from name that is not alphanumeric, space, underscore or hyphen.
        """
        return re.sub(r'[^a-zA-Z0-9 _-]', '', name)

    @staticmethod
    def remove_non_ascii(text) -> str:
        """
        Remove non-ascii characters (to prevent file read errors)
        """
        return ''.join(i for i in text if ord(i) < 128)
