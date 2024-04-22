from genai.client import Client
from genai.credentials import Credentials
from genai.schema import TextGenerationParameters, TextGenerationReturnOptions, DecodingMethod
import os

from dotenv import load_dotenv

load_dotenv()

class WatsonxGenerativeAI:
    def __init__(self):
        # Set your Google Generative AI API key
        self.GENAI_KEY = os.getenv("GENAI_KEY")
        if not self.GENAI_KEY:
            raise ValueError("Missing GENAI_KEY environment variable")

        # Initialize the Generative AI client
        self.client = Client(credentials=Credentials.from_env())

        # Initialize the model object
        self.model = None

    def generate_answer(self, prompt: str, model_name: str = "ibm/granite-13b-instruct-v2"):
        """
        Calls the IBM watsonx Generative API with a given prompt and model name.

        Args:
            prompt (str): The text prompt to send to the model.
            model_name (str, optional): The name of the generative model to use. Defaults to "ibm/granite-13b-instruct-v2".

        Returns:
            dict: A dictionary containing the following keys:
                - "llm": The name of the language model used, prefixed with "ibm/".
                - "prompt": The input prompt.
                - "answer": The generated text response.
                - "generated_token_count": The number of tokens in the generated response.
                - "input_token_count": The number of tokens in the input prompt.
        """
        parameters = TextGenerationParameters(
            decoding_method=DecodingMethod.SAMPLE,
            max_new_tokens=500,
            min_new_tokens=5,
            temperature=0.7,
            top_k=50,
            top_p=1,
            return_options=TextGenerationReturnOptions(input_text=True),
        )

        for response in self.client.text.generation.create(model_id=model_name, 
                        inputs=prompt, 
                        parameters=parameters):
            result = response.results[0].generated_text

        return {
            "llm": model_name,
            "prompt": prompt,
            "answer": result,
            "generated_token_count": response.results[0].generated_token_count,
            "input_token_count": response.results[0].input_token_count
        }


# Example usage
if __name__ == "__main__":
    prompt = "Write a poem about a cat"
    watsonx_ai = WatsonxGenerativeAI()
    generated_text = watsonx_ai.generate_answer(prompt, "ibm/granite-13b-instruct-v2")
    print(generated_text)