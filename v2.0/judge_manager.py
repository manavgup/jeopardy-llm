from typing import List, Tuple
from db_operations import JeopardyDB, LLMResponse
from judges.judge import Judge
from judges.anthropic_claude_judge import AnthropicClaudeJudge
from judges.openai_judge import GPT4Judge
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class JudgeManager:
    def __init__(self, db_file: str, judge_llm: str = ""):
        self.db = JeopardyDB(db_file=db_file)
        self._judges: List[Judge] = self._initialize_judges(judge_llm)

    def _initialize_judges(self, judge_llm: str) -> List[Judge]:
        if not judge_llm:
            return [AnthropicClaudeJudge(env_key="ANTHROPIC_API_KEY"), GPT4Judge(env_key="OPENAI_API_KEY")]
        elif judge_llm.startswith("claude"):
            return [AnthropicClaudeJudge(env_key="ANTHROPIC_API_KEY")]
        elif judge_llm.startswith("gpt-4"):
            return [GPT4Judge(env_key="OPENAI_API_KEY")]
        else:
            raise ValueError(f"Invalid judge_llm: {judge_llm}")

    @property
    def judges(self) -> List[Judge]:
        return self._judges

    def read_llm_responses(self) -> List[Tuple[int, int, LLMResponse]]:
        test_run_id = self.db.get_last_test_run_id()
        llm_responses = self.db.get_llm_responses(test_run_id)
        return [(test_run_id, response.llm_id, response) for response in llm_responses]
   
    def generate_judgements(self, db_file: str, judge_llm: str, llm_id: int, test_run_id: Optional[int] = None):
        judge_manager = JudgeManager(db_file, judge_llm)
        # get llm responses from LLMResponse table where llm_id = llm_id
        if test_run_id is None:
            test_run_id = judge_manager.db.get_last_test_run_id()
        
        llm_responses = judge_manager.db.get_unrated_llm_responses(llm_id, test_run_id, judge_llm)
        for llm_response in llm_responses:
            for judge in judge_manager.judges:
                if judge.total_generated_tokens is not None:
                    judged_response = judge.judge_llmresponse(llm_response, test_run_id)
                    judge_manager.db.insert_llm_judge_rating(judged_response)