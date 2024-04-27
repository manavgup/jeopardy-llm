from judge_manager import JudgeManager
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_file", 
        type=str, 
        default="outs/jeopardy.db", 
        help="The database file to use.")
    parser.add_argument(
        "--judge_llm",
        type=str,
        default=None,
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
        default=None,
        help="The LLM ID to use."
    )
    args = parser.parse_args()
    manager = JudgeManager(args.db_file, args.judge_llm)
    manager.generate_judgements(args.db_file, args.judge_llm, args.llm_id, args.test_run_id)
    
    # judge_manager.run()