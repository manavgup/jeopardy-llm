from typing import Any, Tuple
import os
import re
from openai import OpenAI
import anthropic
from dotenv import load_dotenv
from prompts import PROMPTS
from db_operations import LLMJudgeRating, LLMResponse

class BaseJudge:
    def __init__(self):
        load_dotenv()
        self.name = "BaseJudge"
        self.total_generated_tokens = 0
        self.total_input_tokens = 0

    def judge_response(self, llmresponse: LLMResponse, test_run_id: int) -> LLMJudgeRating:
        ratings = {}
        for evaluation_type in ["is_question", "accuracy", "coherence", "completeness"]:
            ratings[evaluation_type] = self._evaluate(evaluation_type, llmresponse.prompt, llmresponse.response)

        return LLMJudgeRating(
            accuracy=ratings["accuracy"],
            coherence=ratings["coherence"],
            completion=ratings["completeness"],
            question_structure=ratings["is_question"],
            generated_tokens=self.total_generated_tokens,
            input_token_count=self.total_input_tokens,
            llm_response_id=llmresponse.id,
            test_run_id=test_run_id,
            judge_model=self.name
        )

    def _evaluate(self, evaluation_type: str, llm_prompt: str, llm_response: str, max_tokens: int = 7, temperature: float = 0) -> float:
        if evaluation_type == "accuracy":
            user_prompt = PROMPTS[evaluation_type]["user"].format(llm_prompt, llm_response)
        elif evaluation_type == "completeness":
            user_prompt = PROMPTS[evaluation_type]["user"].format(llm_prompt, llm_response)
        else:
            user_prompt = PROMPTS[evaluation_type]["user"].format(llm_response)
        system_prompt = PROMPTS[evaluation_type]["system"]
        input_tokens, generated_tokens, response = self._make_api_call(user_prompt, system_prompt, max_tokens, temperature)
        print(response)
        self.total_generated_tokens += generated_tokens
        self.total_input_tokens += input_tokens

        # convert response to float if it's a number
        try:
            if (type(response) != type(0.1)):
                match = re.search(r"(?:\d+(?:\.\d*)?|\.\d+)", response)
                if match:
                        return float(match.group())
        except ValueError:
            return 0.0
    
    def _make_api_call(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> Tuple[int, int, Any]:
        raise NotImplementedError("_make_api_call method must be implemented in the subclass")

class AnthropicClaudeJudge(BaseJudge):

    def __init__(self):
        super().__init__()
        self.name = "claude-3-opus-20240229"
        self.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.ANTHROPIC_API_KEY)

    def _make_api_call(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> Tuple[int, int, Any]:
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
            return 0, 0, None

class GPT4Judge(BaseJudge):

    def __init__(self):
        super().__init__()
        self.name = "gpt-4"
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        self.openAI = OpenAI(api_key=self.OPENAI_API_KEY)

    def _make_api_call(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> Any:
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
            return (
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
                float(completion.choices[0].message.content)
            )
        except Exception as e:
            print(f"Error making API call: {e}")
            return 0, 0, None

    def _evaluate_accuracy(self, prompt: str, answer: str) -> float:
        # Implement the logic to evaluate the accuracy of the response
        try:
            response = self.openAI.completions.create(
                model=self.name,
                prompt=f"Question: {prompt}\nAnswer: {answer}\nAccuracy score (0-1):",
                max_tokens=1,
                n=1,
                stop=None,
                temperature=0.5,
            )
            accuracy_score = float(response.choices[0].text)
            return accuracy_score
        except Exception as e:
            print(f"Error evaluating accuracy: {e}")
            return 0.00

    # Implement other evaluation methods...

class HumanJudge(BaseJudge):
    name = "Human"

    def _make_api_call(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> Any:
        # Implement the logic to get human evaluation
        accuracy = self._evaluate_accuracy(prompt, system_prompt)
        coherence = self._evaluate_coherence(system_prompt)
        completeness = self._evaluate_completeness(prompt, system_prompt)
        is_question = self._check_if_question(system_prompt)

        return {
            "accuracy": accuracy,
            "coherence": coherence,
            "completeness": completeness,
            "is_question": is_question
        }

    def _evaluate_accuracy(self, prompt: str, answer: str) -> float:
        # Implement the logic to evaluate the accuracy of the response
        return 0.9

    def _evaluate_coherence(self, answer: str) -> float:
        # Implement the logic to evaluate the coherence of the response
        return 0.88

    def _evaluate_completeness(self, prompt: str, answer: str) -> float:
        # Implement the logic to evaluate the completeness of the response
        return 0.8

    def _check_if_question(self, answer: str) -> float:
        # Implement the logic to check if the response is formulated as a question
        return 1.0 if answer.endswith("?") else 0.0