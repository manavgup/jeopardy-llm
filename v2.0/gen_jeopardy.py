from llmproviders import LLMProvider
import json
import multiprocess
from db_operations import JeopardyDB, Question, LLMResponse, LLMJudgeRating, TestRun, LLM
from typing import Generator, Tuple, List, Iterator
from prompts import PROMPTS

from genai.client import Client
from genai.credentials import Credentials

client = Client(credentials=Credentials.from_env())

def prepare_prompt_in_batches(db_file: str, batch_size: int) -> Generator[List[Tuple[Question, str]], None, None]:
    """
    Generate prompts in batches from the questions in the database.

    Args:
        db_file (str): Name of the database file.
        batch_size (int): The number of prompts to generate in each batch.

    Yields:
        List[Tuple[Question, str]]: A list of prompts, where each prompt is a tuple containing the Question object and the prompt text.
    """
    with JeopardyDB(db_file) as db:
        questions = db.get_questions()
        batches = list(chunked_iterable(questions, batch_size))

        for batch in batches:
            prompts = [(question, get_prompt(question)) for question in batch]
            yield prompts

def get_prompt(question: Question):
    """
    Prepare the prompt for the question.
    Args:
        question (Question): A Question object from the database.
    Returns:
        str: The prompt text.
    """
    prompt = PROMPTS["play"]["user"].format(question.category, question.question)
    return prompt

def generate_jeopardy_answer(prompt: str, llm: LLM, provider: LLMProvider) -> LLMResponse:
    """
    Send a question to the LLM and return the LLMResponse object.

    Args:
        prompt (str): The input prompt for the model.
        llm (LLM): The language model object to use.

    Returns:
        LLMResponse: An LLMResponse object containing the generated response.
    """
    result = provider.generate_answer(prompt, llm.provider, llm.name)

    llm_response = LLMResponse(
        llm_id=llm.id,
        response=result['answer'],
        generated_tokens=result['generated_token_count'],
        input_token_count=result['input_token_count']
    )

    return llm_response


def process_chunk(prompts: List[Tuple[Question, str]], 
                  llm: LLM, provider: LLMProvider, test_run_id: int, 
                  db_file: str) -> List[Tuple[Question, LLMResponse]]:
    """Process a chunk of tasks."""
    with JeopardyDB(db_file) as db:
        results = []
        for question, prompt in prompts:
            llm_response = generate_jeopardy_answer(prompt, llm, provider)
            llm_response.question_id = question.id
            llm_response.test_run_id = test_run_id
            llm_response.prompt = prompt
            db.insert_llm_response(llm_response)
            results.append((question, llm_response))
    return results

def generate_answers(llm : LLM, db_file: str, test_run_id: int, batch_size: int):    
    """Generate answers for questions in the database."""
    provider = LLMProvider(llm.provider)
    with multiprocess.Pool() as pool:
        batches = prepare_prompt_in_batches(db_file, batch_size)
        
        args_chunks =  [(batch, llm, provider, test_run_id, db_file) for batch in batches]
        pool.starmap(process_chunk, args_chunks)

def load_json_data(file_path: str):
    """
    Load data from a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def chunked_iterable(iterable, size):
    """
    Yield successive size chunks from iterable.
    """
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def main(models_file: str, questions_file: str, db_file: str):
    """
    Main execution function.
    """
    # Load LLM information from the models file and insert them into the database
    with JeopardyDB(db_file) as db:
        llms = db.insert_and_return_llms_from_file(models_file)
        # Load the questions from the file and insert them into the database
        db.insert_questions_file(questions_file)
        test_run_id = db.insert_test_run(PROMPTS["play"]["user"], PROMPTS["play"]["system"])

    for llm in llms:
        generate_answers(llm, db_file, test_run_id, 25)   # batch size of 25

    
if __name__ == "__main__":
    models_file = "data/models.jsonl"  # Path to the models configuration file
    questions_file = "data/questions-test.jsonl"  # Path to the questions file
    db_file = "outs/jeopardy.db"
    main(models_file, questions_file, db_file)

