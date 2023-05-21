import sys


class ConsoleLogger:
    """
    This class handles console logging in various colors consistently across the project.
    Additionally, handles coloring streamed output from the Language Models, which is only available on select LLMs
    """

    # ANSI color codes
    COLOR_CYAN = "\033[96m"
    COLOR_MAGENTA = "\033[95m"
    COLOR_GREEN = "\033[92m"
    COLOR_RED = "\033[91m"
    COLOR_RESET = "\033[0m"

    COLOR_INPUT = COLOR_CYAN
    COLOR_THINKING = COLOR_MAGENTA
    COLOR_REPSONSE = COLOR_GREEN
    COLOR_TOOL = COLOR_MAGENTA
    COLOR_ERROR = COLOR_RED

    # Controls the color of streamed output
    current_stream_color = COLOR_RESET

    #region Logging Methods
    # - These methods are used to control the color of output & input

    @staticmethod
    def log(message, color, prefix=None):
        if prefix:
            print(f"{color}{prefix}: {message}{ConsoleLogger.COLOR_RESET}")
        else:
            print(f"{color}{message}{ConsoleLogger.COLOR_RESET}")

    @staticmethod
    def input(message, color=COLOR_INPUT):
        user_input = input(f"{ConsoleLogger.COLOR_CYAN}{message}")
        return user_input

    @staticmethod
    def log_input(input_text):
        print(f"{ConsoleLogger.COLOR_INPUT}INPUT: {input_text}{ConsoleLogger.COLOR_RESET}")

    @staticmethod
    def log_thinking():
        print(f"\n{ConsoleLogger.COLOR_THINKING}Thinking...{ConsoleLogger.COLOR_RESET}\n")

    @staticmethod
    def log_response(response_text):
        print(f"{ConsoleLogger.COLOR_REPSONSE}RESPONSE: {response_text}{ConsoleLogger.COLOR_RESET}")

    @staticmethod
    def log_tool(tool_text):
        print(f"{ConsoleLogger.COLOR_TOOL}TOOL: {tool_text}{ConsoleLogger.COLOR_RESET}")

    @staticmethod
    def log_error(error_text):
        print(f"{ConsoleLogger.COLOR_ERROR}ERROR: {error_text}{ConsoleLogger.COLOR_RESET}")

    #endregion
    #region Streamed output
    # - These methods are used in custom_stream to control the color of streamed output, which is only available on the paid OpenAI API (as far as I know).

    @staticmethod
    def log_streaming(token: str):
        sys.stdout.write(ConsoleLogger.current_stream_color + token + ConsoleLogger.COLOR_RESET)
        sys.stdout.flush()

    @staticmethod
    def set_stream_color(color: str):
        ConsoleLogger.current_stream_color = color

    @staticmethod
    def set_response_stream_color():
        ConsoleLogger.set_stream_color(ConsoleLogger.COLOR_REPSONSE)

    @staticmethod
    def set_default_stream_color():
        ConsoleLogger.set_stream_color(ConsoleLogger.COLOR_RESET)

    #endregion
