import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

class AnthropicClaude:
    def __init__(self):
        # Set your Anthropic API key
        self.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
        
        # Initialize the Anthropic AI client
        self.client = anthropic.Anthropic(api_key=self.ANTHROPIC_API_KEY)

    def generate_answer(self, prompt: str, model_name: str):
        """
        Send a question to the Claude model and return the response.

        Args:
            prompt (str): The input prompt for the model.
            model_name (str): The name of the Claude model to use.

        Returns:
            dict: A dictionary containing the following keys:
                - "llm": The name of the language model used, prefixed with "claude/".
                - "prompt": The input prompt.
                - "answer": The generated text response.
                - "generated_token_count": The number of tokens in the generated response.
                - "input_token_count": The number of tokens in the input prompt.
        """
        # Define the instruction for the Claude model
        instruction = {
            "role": "assistant",
            "content": "You are Claude, an AI assistant created by Anthropic. Please provide a helpful response to the user's query."
        }

        # Format the prompt as expected by the API
        formatted_prompt = f"\n\nHuman: {prompt}"

        # Call the Claude model and get the response
        response = self.client.messages.create(
            model=model_name,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        # Calculate the token counts
        return {
            "llm": f"claude/{model_name}",
            "prompt": prompt,
            "answer": response.content[0].text,
            "generated_token_count": response.usage.output_tokens,
            "input_token_count": response.usage.input_tokens
        }

# Example usage
if __name__ == "__main__":
    prompt = "Who killed Jessica Rabbit?"
    claude = AnthropicClaude()
    generated_text = claude.generate_answer(prompt, 'anthropic', "claude-3-opus-20240229")
    print(generated_text)