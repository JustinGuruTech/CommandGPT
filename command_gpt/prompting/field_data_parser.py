# This file contains the InitialParser class, which is used to parse input into key 
# components that are relevant to Crewtracks

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from command_gpt.utils.console_logger import ConsoleLogger
from config import default_llm_open_ai

class FieldDataParser:
    """
    This class parses input into key components that are relevant to Crewtracks
    """

    def generate_prompt_string(self, input_data) -> str:
        """
        Generates a prompt string from the current state of the parser
        :return: A string that can be used as a prompt
        """

        key_data_models = "Job, Date, Time, Location, Crew, Equipment, Materials, Notes, Hours, Quantity, Total, Cost"

        # Initialize prompt string
        prompt_string = "You are FieldDataParser-GPT, an AI designed to parse input into key components that are relevant to field tracking, keeping in mind that the user base is not always the most tech savvy. "
        prompt_string += "The output of your work will be provided to FieldDataMapper-GPT, an AI designed to map semi-structured field tracking data to the correct fields in the database. It is critical that you parse and provide details correctly, as any mistakes will be passed to FieldDataMapper-GPT and result in data corruption. "
        prompt_string += "In order to parse information, you should break it down into a list of information that is relevant to field tracking. Represent each piece of information as a single line, ensuring each piece of information is included exactly once, and that the order of information is preserved. Ensure no hallucinations. "
        prompt_string += "This is a semi-conversational process, so you should ask questions and provide feedback to the user as needed. Due to limited context, ensure your responses are on topic and helpful. "
        prompt_string += "You should also provide a summary of the information you have parsed, and ask the user to confirm that it is correct. If the user indicates that the information is incorrect, you should ask for clarification and attempt to correct it. "
        prompt_string += "when you feel you have successfully parsed the data, provide a brief overview and your confidence in your parsing of the data, followed by the data, line by line as described above. "
        prompt_string += "\n\nPlease use the following key data models -- " + key_data_models + " -- Here is the current user input: " + input_data + "\n\n"
        return prompt_string
        
    def run(self) -> str:
        """
        Kicks off interaction loop with AI
        """

        ConsoleLogger.log("\nKicking off the field data parser...)\n", color=ConsoleLogger.COLOR_MAGENTA)

        # Get input from user
        input_field_data = ConsoleLogger.input("Provide your field data to parse (type 'exit' to quit): ")
    
        # Generate the prompt string
        prompt_string = self.generate_prompt_string(input_field_data)

        # Initialize the conversation chain
        llm = default_llm_open_ai
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt_string),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        memory = ConversationBufferMemory(return_messages=True)
        conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)

        # Get initial parsing response from AI
        response = conversation.predict(input=input_field_data)

        # Interaction Loop
        loop_count = 0
        while True:
            input_data = ConsoleLogger.input("Provide any feedback or type 'y' to accept and continue (type 'exit' to quit): ")

            # Increment loop count
            loop_count += 1

            # Exit if the user indicates they are done
            if input_data == "exit":
                return "parser exited. Last output:\n\n" + response
            elif input_data == "y":
                ConsoleLogger.log("Field data parsing complete!", color=ConsoleLogger.COLOR_MAGENTA)
                return response
            else:
                response = conversation.predict(input=input_data)
                