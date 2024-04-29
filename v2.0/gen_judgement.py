from typing import Dict, List, Tuple
from llm_judgement import BaseJudge, AnthropicClaudeJudge, GPT4Judge, HumanJudge
from db_operations import JeopardyDB, LLMResponse, LLMJudgeRating
import argparse
import re

class JudgeManager:
    def __init__(self, db_file: str, judge_llm: str = None):
        self.db = JeopardyDB(db_file=db_file)
        if judge_llm is None:
            self.judges: List[BaseJudge] = [ AnthropicClaudeJudge(), GPT4Judge()]
        elif judge_llm.startswith("claude"):
            self.judges: List[BaseJudge] = [AnthropicClaudeJudge()]
        elif judge_llm.startswith("gpt-4"):
            self.judges: List[BaseJudge] = [GPT4Judge()]
        
        

    def read_llm_responses(self) -> List[Tuple[int, int, LLMResponse]]:
        """
        Read the LLM responses from the database.
        Returns:
            List[Tuple[int, int, LLMResponse]]: A list of tuples containing the test run ID, LLM ID, and LLMResponse object.
        """
        test_run_id = self.db.get_last_test_run_id()
        llm_responses = self.db.get_llm_responses(test_run_id)
        print(llm_responses)
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
    
def generate_judgements(db_file: str, judge_llm: str, llm_id: int, test_run_id: int = None):
    judge_manager = JudgeManager(db_file, judge_llm)
    # get llm responses from LLMResponse table where llm_id = llm_id
    if test_run_id is None:
        test_run_id = judge_manager.db.get_last_test_run_id()
    llm_responses = judge_manager.db.get_unrated_llm_responses(llm_id, test_run_id, judge_llm)
    for llm_response in llm_responses:
        for judge in judge_manager.judges:
            if judge.total_generated_tokens is not None:
                judged_response = judge.judge_response(llm_response, test_run_id)
                judge_manager.db.insert_llm_judge_rating(judged_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_file", 
        type=str, 
        default="output/jeopardy.db", 
        help="The database file to use.")
    parser.add_argument(
        "--judge_llm",
        type=str,
        default= None,
        help="Name of the judge LLM (e.g.,'claude-3-opus-20240229', 'gpt-4')"
    )
    parser.add_argument(
        "--test_run_id",
        type=int,
        default=None,
        help="The test run ID to use.")
    parser.add_argument(
        "--llm_id",
        type=int,
        default=3,
        help="The LLM ID to use."
    )
    args = parser.parse_args()
    generate_judgements(args.db_file, args.judge_llm, args.llm_id, args.test_run_id)
    
    # JudgeManager.run()