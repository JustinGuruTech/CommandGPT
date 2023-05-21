# This file contains some methods for prompting the user and wrapping the input in another prompt that will generate a ruleset for CommandGPT.

from langchain import ConversationChain, PromptTemplate

from config import default_llm_open_ai
from command_gpt.utils.console_logger import ConsoleLogger

def prompt_for_ruleset(ruleset_that_will: str, user_prompt: str) -> str:
    """
    Builds a nesting doll of prompts that ultimately generates a ruleset for CommandGPT.
    :param ruleset_that_will: Overall goal of rule set (research, event planning, law, etc.).
    :param user_prompt: The input to the ruleset.

    This is a nesting doll of prompts, generally structured like this:
    1. Contextualize the LLM on CommandGPT in a generally applicable way.
    2. Insert the request: "Generate an input summary that will instruct CommandGPT to {request}:"
    3. Prompt the user for a specific topic, which will be used to generate the ruleset.
    """
    request_template = f"CommandGPT is an LLM driven application that operates in a continuous manner to achieve specified goals and fulfill a given purpose. CommandGPT uses commands to interface with it's virtual environment.\n\nThe goal of this project is to specify an initial rule set that keeps CommandGPT on track autonomously. Ideally, CommandGPT will improve it's strategy along the way without losing context on the information it has already gathered, but this can be challenging.\n\nIt's important that AutoGPT writes information into files and folders as it conducts research, but it needs to be specifically instructed to do so else it will default to internally processing.\n\nThe ruleset should: \n- be precise & descriptive, structured in a way that is easy to understand and act on for an LLM. \n- be phrased as \"You are xxx-gpt (filling in a descriptive & relevant name), an AI designed to...\" \n- should be followed by 5 Goals: These should break down the input into descriptive and well balanced topics.\nGenerate an input summary that will instruct CommandGPT to {ruleset_that_will}:\n\n{{topic}}\n\nEnsure that the output includes only a Name, Purpose, and 5 Goals that will act as instructions for CommandGPT. Ensure the output is formatted as \"You are xxxx-gpt, an AI designed to<purpose>\"\nGoals:\n1. <goal 1>\n2. <goal 2>\n3. <goal 3>\n4. <goal 4>\n5. <goal 5>\n\n"

    request_prompt_template = PromptTemplate(
        input_variables=["topic"],
        template=request_template
    )

    # Initialize conversation with LLM
    conversation = ConversationChain(
        llm=default_llm_open_ai,
    )

    # Get research topic from user and format prompt
    topic = ConsoleLogger.input(f"{user_prompt}: ")
    formatted_prompt = request_prompt_template.format(
        topic=topic
    )
    ConsoleLogger.log_input(formatted_prompt)

    # Start the conversation
    ConsoleLogger.set_response_stream_color() # Set color for response
    response = conversation.predict(input=formatted_prompt)
    return response

def prompt_for_research_ruleset():
    """
    Uses prompt_for_ruleset() to prompt the user for a topic in order to generate a ruleset for CommandGPT that will conduct research on the topic.
    """
    # Get research topic from user and format prompt
    return prompt_for_ruleset(
        ruleset_that_will="conduct research and generate reports on",
        user_prompt="Enter a research topic"
    )

def prompt_for_event_planning_ruleset():
    """
    Uses prompt_for_ruleset() to prompt the user for an event in order to generate a ruleset for CommandGPT that will plan the event.
    """
    # Get event from user and format prompt
    return prompt_for_ruleset(
        ruleset_that_will="plan an event for",
        user_prompt="Enter an event"
    )