from typing import Tuple
from openai import OpenAI

from judges.judge import Judge
import os

class GPT4Judge(Judge):
    def __init__(self, env_key_project: str="OPENAI_PROJECT_ID", env_key_org: str="OPENAI_ORG_ID"):
        super().__init__(name="gpt-4", env_key="")
        self.project_key = os.environ.get(env_key_project)
        self.org_id = os.environ.get(env_key_org)
        self.openAI = OpenAI(organization=self.org_id, project=self.project_key)

    def _make_api_call(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> Tuple[int, int, str]:
        try:
            completion = self.openAI.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if completion.usage:
                prompt_tokens = completion.usage.prompt_tokens if completion.usage.prompt_tokens is not None else 0
                completion_tokens = completion.usage.completion_tokens if completion.usage.completion_tokens is not None else 0
            else:
                prompt_tokens = 0
                completion_tokens = 0
                
            content = completion.choices[0].message.content if completion.choices[0].message.content is not None else ""
            return (prompt_tokens, completion_tokens, content)
        except Exception as e:
            print(f"Error making API call: {e}")
            return (0, 0, "")

