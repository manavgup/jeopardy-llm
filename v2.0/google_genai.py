import google.generativeai as genai
from genai.client import Client
from genai.credentials import Credentials
import os

from dotenv import load_dotenv

load_dotenv()

class GoogleGenerativeAI:
    def __init__(self):
        # Set your Google Generative AI API key
        self.GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
        if not self.GOOGLE_GENAI_API_KEY:
            raise ValueError("Missing GOOGLE_GENAI_API_KEY environment variable")

        # Initialize the Generative AI client
        self.client = Client(credentials=Credentials.from_env())

        # Initialize the model object
        self.model = None

    def generate_answer(self, prompt: str, model_name: str = "gemini-1.0-pro-latest"):
        """
        Calls the Google Generative API with a given prompt and model name.

        Args:
            prompt (str): The text prompt to send to the model.
            model_name (str, optional): The name of the generative model to use. Defaults to "gemini-1.0-pro-latest".

        Returns:
            dict: A dictionary containing the following keys:
                - "llm": The name of the language model used, prefixed with "google/".
                - "prompt": The input prompt.
                - "answer": The generated text response.
                - "generated_token_count": The number of tokens in the generated response.
                - "input_token_count": The number of tokens in the input prompt.
        """
        # Configure the API key only once (on first call)
        genai.configure(api_key=self.GOOGLE_GENAI_API_KEY)

        # Create the model object only once (on first call for a specific model)
        if not self.model or self.model.name != model_name:
            self.model = genai.GenerativeModel(model_name)

        # Generate content
        response = self.model.generate_content(prompt)

        # Calculate the token counts
        input_token_count = self.model.count_tokens(prompt).total_tokens
        generated_token_count = self.model.count_tokens(response.text).total_tokens

        return {
            "llm": f"google/{model_name}",
            "prompt": prompt,
            "answer": response.text,
            "generated_token_count": generated_token_count,
            "input_token_count": input_token_count
        }

# Example usage
if __name__ == "__main__":
    prompt = "Write a poem about a cat"
    google_ai = GoogleGenerativeAI()
    generated_text = google_ai.generate_answer(prompt, "google", "gemini-1.0-pro-latest")
    print(generated_text)