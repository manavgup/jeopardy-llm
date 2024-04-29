from typing import Tuple
import anthropic
from judges.judge import Judge

class AnthropicClaudeJudge(Judge):
    def __init__(self, env_key: str = "ANTHROPIC_API_KEY"):
        super().__init__(name="claude-3-opus-20240229", env_key=env_key)
        #self.name = "claude-3-opus-20240229"
        self.client = anthropic.Anthropic(api_key=self.api_key)
              
    def _make_api_call(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> Tuple[int, int, str]:
        try:
            response = self.client.messages.create(
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return ( 
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.content[0].text
            )
        except Exception as e:
            print(f"Error making API call: {e}")
            return 0, 0, ""