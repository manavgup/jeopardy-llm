from typing import Dict, List, Tuple
from llm_judgement import BaseJudge, AnthropicClaudeJudge, GPT4Judge, HumanJudge
from db_operations import JeopardyDB, LLMResponse, LLMJudgeRating

class JudgeManager:
    def __init__(self, db_file: str):
        self.db = JeopardyDB(db_file=db_file)
        self.judges: List[BaseJudge] = [AnthropicClaudeJudge(), GPT4Judge(), HumanJudge()]

    def read_llm_responses(self) -> List[Tuple[int, int, LLMResponse]]:
        """
        Read the LLM responses from the database.
        Returns:
            List[Tuple[int, int, LLMResponse]]: A list of tuples containing the test run ID, LLM ID, and LLMResponse object.
        """
        test_run_id = self.db.get_last_test_run_id()
        llm_responses = self.db.get_llm_responses(test_run_id)
        return [(test_run_id, response.llm_id, response) for response in llm_responses]

    def judge_response(self, llm_response: LLMResponse) -> 'LLMJudgeRating':
        """
        Judge an LLM response and store the results in the database.
        Args:
            llm_response (LLMResponse): The LLMResponse object to judge.
        Returns:
            LLMJudgeRating: The judged response.
        """
        judged_response = LLMJudgeRating()
        judged_response.llm_response_id = llm_response.id
        judged_response.test_run_id = llm_response.test_run_id
        judged_response.accuracy = 0.0
        for judge in self.judges:
            judged_response["ratings"][judge.name] = judge.judge_response(llm_response.prompt, llm_response.response)
            self.db.insert_llm_judge_rating(
                llm_response_id=llm_response.id,
                accuracy=judged_response["ratings"][judge.name]["accuracy"],
                coherence=judged_response["ratings"][judge.name]["coherence"],
                completion=judged_response["ratings"][judge.name]["completeness"],
                question_structure=judged_response["ratings"][judge.name]["is_question"]
            )
        return judged_response
    
    def judge_responses(self, responses: List[Tuple[int, int, LLMResponse]]) -> None:
        """
        Judge the LLM responses and store the results in the database.
        Args:
            responses (List[Tuple[int, int, LLMResponse]]): A list of tuples containing the test run ID, LLM ID, and LLMResponse object.
        """
        for test_run_id, llm_id, response in responses:
            judged_response = {
                "prompt": response.prompt,
                "answer": response.response,
                "ratings": {}
            }
            for judge in self.judges:
                judged_response["ratings"][judge.name] = judge.judge_response(response.prompt, response.response)
                self.db.insert_llm_judge_rating(
                    llm_response_id=response.id,
                    accuracy=judged_response["ratings"][judge.name]["accuracy"],
                    coherence=judged_response["ratings"][judge.name]["coherence"],
                    completion=judged_response["ratings"][judge.name]["completeness"],
                    question_structure=judged_response["ratings"][judge.name]["is_question"]
                )

    def run(self):
        responses = self.read_llm_responses()
        self.judge_responses(responses)


if __name__ == "__main__":
    judge_manager = JudgeManager(db_file="outs/jeopardy.db")

    test_run_id = judge_manager.db.get_last_test_run_id()
    llm_responses = judge_manager.db.get_llm_responses(test_run_id)
    for llm_response in llm_responses:
        for judge in judge_manager.judges:
            judged_response = judge.judge_response(llm_response)
            judge_manager.db.insert_llm_judge_rating(judged_response)
    # judge_manager.run()