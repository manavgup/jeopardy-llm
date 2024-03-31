import sqlite3
import csv
import os
import re
from dotenv import load_dotenv
from openai import OpenAI


from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    DecodingMethod,
    LengthPenalty,
    SystemMessage,
    TextGenerationParameters,
    TextGenerationReturnOptions,
)
from genai.text.generation import CreateExecutionOptions

# Replace 'your_database.db' with your actual database path
DATABASE_PATH = './jeopardy.db'
CSV_FILE_PATH = './dataset.csv'
load_dotenv()
client = Client(credentials=Credentials.from_env())
openAI = OpenAI()

# RESET FLAG
RESET_FLAG = False

# List of LLMs to use
llms = ['ibm/granite-13b-chat-v1',
        'ibm/granite-13b-chat-v2',
        'meta-llama/llama-2-70b',
        'ibm-mistralai/mixtral-8x7b-instruct-v01-q']

JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user for playing the game of Jeopardy.
You will be provided the category, statement, and the response received from the AI assistant. 
Your evaluation should consider the rules of Jeopardy game such as the accuracy , and formulation of the response in the form of a question. 
Be as objective as possible. 
You must rate the response on a scale of 1 to 5 where 5 is the most accurate, and in the form of a question.

Your response must strictly following this format: 

Feedback:::
Rating: (your rating, as a float between 1 and 5).

Now here are the category, statement, and the response.

Category: {category}
Statement: {question}
Answer: {answer}

Please provide your rating for the response provided by the AI assistant.

Feedback:::
Rating: """

parameters = TextGenerationParameters(
    max_new_tokens=500,
    decoding_method=DecodingMethod.SAMPLE,
    length_penalty=LengthPenalty(start_index=5, decay_factor=1.5),
    return_options=TextGenerationReturnOptions(
        # if ordered is False, you can use return_options to retrieve the corresponding prompt
        input_text=True,
    ),
)

def create_connection(db_file):
    """Create a database connection to a SQLite database"""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn

def create_tables(conn):
    """Create the table if it doesn't already exist"""
    create_table_questions = """
    CREATE TABLE IF NOT EXISTS questions (
        question_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category VARCHAR(100),
        question VARCHAR(2000),
        expected_answer VARCHAR(2000),
        llm_answer VARCHAR(2000)
    );
    """
    create_table_llms = """
    CREATE TABLE IF NOT EXISTS llms (
        llm_id INTEGER PRIMARY KEY AUTOINCREMENT,
        llm_name VARCHAR(100),
        prompt VARCHAR(2000)
    );
    """
    create_table_results = """
    CREATE TABLE IF NOT EXISTS results (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        question_id INTEGER,
        llm_id INTEGER,
        run_id INTEGER,
        result VARCHAR(100),
        gpt_rating INTEGER,
        FOREIGN KEY(question_id) REFERENCES questions(question_id),
        FOREIGN KEY(llm_id) REFERENCES llms(llm_id)
    );"""

    try:
        c = conn.cursor()
        c.execute(create_table_questions)
        c.execute(create_table_llms)
        c.execute(create_table_results)
    except sqlite3.Error as e:
        print(e)

def get_run_id(file_path='run_id.txt'):
    try:
        with open(file_path, 'r') as file:
            last_run_id = int(file.read().strip())
            current_run_id = last_run_id + 1
    except FileNotFoundError:
        current_run_id = 1
    
    with open(file_path, 'w') as file:
        file.write(str(current_run_id))
    
    return current_run_id

def insert_llms(conn, llms):
    """Insert the LLMs into the database"""
    # check if llms database is empty first
    cur = conn.cursor()
    cur.execute("SELECT EXISTS(SELECT 1 FROM llms)")
    if cur.fetchone()[0] == 1:
        return
    
    for llm in llms:
        sql = ''' INSERT INTO llms(llm_name)
                  VALUES(?) '''
        cur = conn.cursor()
        cur.execute(sql, (llm,))
        conn.commit()

def check_data_exists(conn):
    """Check if any data exists in the questions table"""
    cur = conn.cursor()
    cur.execute("SELECT EXISTS(SELECT 1 FROM questions)")
    return cur.fetchone()[0] == 1

def load_data_from_csv(conn, csv_file_path):
    """Load questions and answers from a CSV file into the database."""
    with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            question_id = row['question_id']
            category = row['category']
            question = row['question']
            expected_answer = row['answer']
            # llm_answer and result are not available yet, so we insert NULLs
            sql = '''INSERT INTO questions(question_id, category, question, expected_answer)
                     VALUES(?,?,?,?)'''
            cur = conn.cursor()
            cur.execute(sql, (question_id, category, question, expected_answer))
            conn.commit()

def get_unanswered_questions(conn):
    """Query all questions that haven't been answered by the LLM yet"""
    cur = conn.cursor()
    cur.execute("SELECT question_id, category, question FROM questions WHERE llm_answer IS NULL")
    
    return cur.fetchall()

def update_results(conn, run_id, llm_id, question_id, llm_answer):

    # enumerate through the questions and llm_answers
    # find corresponding llm_answer
    # insert the llm_answer into the results database
    sql = ''' INSERT INTO results(run_id, question_id, llm_id, result)
                VALUES(?, ?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, (run_id, question_id, llm_id, llm_answer))
    conn.commit()

def generate_jeopardy_prompts(llm, question_id, category, question):
    """Generate prompts for the LLM to play Jeopardy"""
    prompt_ids = []
    parameters = TextGenerationParameters(
        decoding_method=DecodingMethod.SAMPLE, 
        max_new_tokens=500, 
        min_new_tokens=5, 
        temperature=0.7, 
        top_k=50, 
        top_p=1,
        return_options=TextGenerationReturnOptions(input_text=True),
    )

    prompt = f"You are playing the game of jeopardy. You will be given a category and a statement." \
            f"Using the information provided, you must respond with a question." \
            f"Category: '{category}'. Here's your statement: '{question}'. What's your response?"

    create_response = client.prompt.create(
                        name=f"jp_{question_id}",
                        model_id=llm,
                        input=prompt,
                        parameters=parameters)
    return create_response.result.id

def ask_llm(llm, question, prompt_id):
    """Send a question to the LLM and return the response"""
    result = None

    for response in client.text.generation.create(model_id=llm, 
                    prompt_id=prompt_id,
                    inputs=question, 
                    parameters=parameters):
         result = response.results[0].generated_text

    return result

def rate_via_chatGPT(conn, category, question, answer):
    # method to use chatGPT to evaluate the answers
    completion = openAI.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": JUDGE_PROMPT.format(category=category, question=question, answer=answer)}
                ]
            )
    # return the results
    return completion.choices[0].message.content

def update_gpt_rating(conn, question_id, llm_id, gpt_rating):
    # update the gpt_rating in the database
    cur = conn.cursor()
    cur.execute("UPDATE results SET gpt_rating = ? WHERE question_id = ? AND llm_id = ?",
                (gpt_rating, question_id, llm_id))
    conn.commit()

# write a method that parses the response from chatGPT and extracts the rating
# the rating is found in the response after the "Rating: " string
def extract_rating(response: str, split_str="Rating: ") -> int:
    try:
        if split_str in response:
            rating = response.split(split_str)[1]
        else:
            rating = response
        digits = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digits[0])
    except Exception as e:
        print(f"Error extracting rating: {e}")
        return None

def main():
    # Establish a database connection
    run_id = get_run_id("./.state")
    conn = create_connection(DATABASE_PATH)
    print ("Opened database successfully")

    if RESET_FLAG:
        print("Resetting database...")
        cur = conn.cursor()
        cur.execute("DELETE FROM questions")
        conn.commit()
        print("Database reset.")

    if conn is not None:
        # Create table
        create_tables(conn)  # Ensure the table is created before checking for data or inserting new data

        if not check_data_exists(conn):
            load_data_from_csv(conn, CSV_FILE_PATH)
            insert_llms(conn, llms)
        else:
            print("Data already exists in the database.")

        # Fetch unanswered questions
        unanswered_questions = get_unanswered_questions(conn)
        for llm_id, llm in enumerate(llms):
            for question_id, category, question in unanswered_questions:
                prompt_id = generate_jeopardy_prompts(llm, question_id, category, question)
                llm_answer = ask_llm(llm, question, prompt_id)

                update_results(conn, run_id, llm_id, question_id, llm_answer)
                # rate_via_chatGPT(conn, llm_answers)
                gpt_rating = extract_rating(rate_via_chatGPT(conn, category, question, llm_answer))
                print(f"Rating for question {question_id} from LLM {llm_id}: {gpt_rating}")
                # Update the database with the GPT rating
                update_gpt_rating(conn, question_id, llm_id, gpt_rating)

        # Process each question
        conn.close()
    else:
        print("Error! cannot create the database connection.")

if __name__ == '__main__':
    main()
