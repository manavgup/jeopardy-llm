import json
import anthropic_genai, google_genai, watsonx_genai
from typing import List, Dict, Any, Union

class LLMProvider:
    def __init__(self, llm: str):
        self.providers: Dict[str] = {}
        self.providers[llm.lower()] = self._initialize_provider(llm.lower())

    def _initialize_provider(self, llm: str) -> 'LLMProvider':
        provider_name = llm.lower()
        if provider_name == "anthropic":
            return anthropic_genai.AnthropicClaude()
        elif provider_name == "google":
            return google_genai.GoogleGenerativeAI()
        elif provider_name == "ibm":
            return watsonx_genai.WatsonxGenerativeAI()
        else:
            raise ValueError(f"Unsupported provider: {llm['provider']}")

    def get_provider(self, provider_name: str) -> 'LLMProvider':
        provider = self.providers[provider_name.lower()]
        if not provider:
            raise ValueError(f"Invalid provider: {provider_name}")
        return provider

    def generate_answer(self, prompt: str, provider_name: str, model: str) -> str:
        return self.get_provider(provider_name).generate_answer(prompt, model)
    
if __name__ == "__main__":
# Example usage
    llms = [
                {'provider': 'Anthropic', 'model': 'claude-3-opus-20240229'},
                {'provider': 'Google', 'model': 'gemini-1.0-pro-latest'},
                {'provider': 'IBM', 'model': 'meta-llama/llama-2-70b'},
                {'provider': 'IBM', 'model': 'ibm/granite-13b-instruct-v2'}
            ]
    for llm in llms:
        llm_provider = LLMProvider(llm['provider'])
        print(llm_provider.generate_answer("Who killed Jessica Rabbit?", llm['provider'], llm['model']))
        print("\n")
    
