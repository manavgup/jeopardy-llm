from typing import Tuple

from judges.judge import Judge

class HumanJudge(Judge):
    name = "Human"

    def _make_api_call(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> Tuple[int, int, str]:
        # Implement the logic to get human evaluation
        return 0,0,""

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

    # Implement the judge_response method