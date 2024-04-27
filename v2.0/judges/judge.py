from typing import Tuple, Optional
from abc import ABC
from dataclasses import dataclass

from db_operations import LLMResponse, LLMJudgeRating
from prompts import PROMPTS
import re
import os

@dataclass
class Judge(ABC):
    name: str
    env_key: str
    total_generated_tokens: int = 0
    total_input_tokens: int = 0
    
    def __post_init__(self):
        self.api_key = os.environ.get(self.env_key)

    def judge_llmresponse(self, llmresponse: LLMResponse, test_id: Optional[int] = None) -> LLMJudgeRating:
        ratings = {}
        for evaluation_type in ["completeness", "accuracy", "coherence", "is_question"]:
            ratings[evaluation_type] = self._evaluate(evaluation_type, llmresponse.prompt, llmresponse.response)

        return LLMJudgeRating(
            accuracy=ratings["accuracy"],
            coherence=ratings["coherence"],
            completion=ratings["completeness"],
            question_structure=ratings["is_question"],
            generated_tokens=self.total_generated_tokens,
            input_token_count=self.total_input_tokens,
            llm_response_id=llmresponse.id,
            test_run_id=test_id,
            judge_model=self.name
        )

    def _evaluate(self, evaluation_type: str, llm_prompt: str, llm_response: str, max_tokens: int = 7, temperature: float = 0) -> float:
        if evaluation_type == "accuracy":
            user_prompt = PROMPTS[evaluation_type]["user"].format(llm_prompt, llm_response)
        elif evaluation_type == "completeness":
            user_prompt = PROMPTS[evaluation_type]["user"].format(llm_response, llm_prompt)
        else:
            user_prompt = PROMPTS[evaluation_type]["user"].format(llm_response)
        system_prompt = PROMPTS[evaluation_type]["system"]
        input_tokens, generated_tokens, response = self._make_api_call(user_prompt, system_prompt, max_tokens, temperature)
        self.total_generated_tokens += generated_tokens
        self.total_input_tokens += input_tokens

        # convert response to float if it's a number
        try:
            match = re.search(r"(?:\d+(?:\.\d*)?|\.\d+)", response)
            if match:
                return float(match.group())
        except ValueError:
            return 0.0
        return 0.0

    def _make_api_call(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> Tuple[int, int, str]:
        raise NotImplementedError("Subclasses must implement the _make_api_call method.")

