from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, text
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import json

Base = declarative_base()

class LLM(Base):
    __tablename__ = 'llms'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    provider = Column(String)

    llm_responses = relationship("LLMResponse", backref="llm")

class Question(Base):
    __tablename__ = 'questions'

    id = Column(Integer, primary_key=True)
    category = Column(String)
    air_date = Column(String)
    question = Column(String)
    value = Column(String)
    answer = Column(String)
    round = Column(String)
    show_number = Column(String)

    llm_responses = relationship("LLMResponse", backref="question")

class LLMResponse(Base):
    __tablename__ = 'llm_responses'

    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey('questions.id'))
    llm_id = Column(Integer, ForeignKey('llms.id'))
    test_run_id = Column(Integer, ForeignKey('test_runs.id'))
    prompt = Column(String)
    response = Column(String)
    generated_tokens = Column(Integer)
    input_token_count = Column(Integer)

    llm_judge_ratings = relationship("LLMJudgeRating", backref="llm_response")

class LLMJudgeRating(Base):
    __tablename__ = 'llm_judge_ratings'

    id = Column(Integer, primary_key=True)
    llm_response_id = Column(Integer, ForeignKey('llm_responses.id'))
    test_run_id = Column(Integer, ForeignKey('test_runs.id'))
    accuracy = Column(Float)
    coherence = Column(Float)
    completion = Column(Float)
    question_structure = Column(Float)
    generated_tokens = Column(Integer)
    input_token_count = Column(Integer)
    judge_model = Column(String)
    judge_llm_response = Column(String)

class TestRun(Base):
    __tablename__ = 'test_runs'

    id = Column(Integer, primary_key=True)
    user_prompt = Column(String)
    system_prompt = Column(String)
    parameters = Column(String)
    run_time = Column(DateTime, default=datetime.now)

    llm_responses = relationship("LLMResponse", backref="test_run")
    llm_judge_ratings = relationship("LLMJudgeRating", backref="test_run")    

class JeopardyDB:
    def __init__(self, db_file='jeopardy.db'):
        self.engine = create_engine(f'sqlite:///{db_file}')
        Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.session = Session()
        self.create_tables()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    def create_tables(self):
        """
        Create the database tables if they don't already exist.
        """
        Base.metadata.create_all(self.engine)

    def insert_llm(self, name, provider):
        llm = LLM(
            name=name,
            provider=provider
        )
        self.session.add(llm)
        self.session.commit()
        return llm.id
    
    def insert_and_return_llms_from_file(self, file_path):
        llms_data = self.load_file(file_path)
        llms = [LLM(name=llm['model'], provider=llm['provider']) for llm in llms_data]
        self.session.add_all(llms)
        self.session.commit()
        return llms

    def get_all_llms(self):
        return self.session.query(LLM).all()

    def insert_question(self, data):
        question = Question(
            category=data['category'],
            air_date=data['air_date'],
            question=data['question'],
            value=data['value'],
            answer=data['answer'],
            round=data['round'],
            show_number=data['show_number']
        )
        self.session.add(question)
        self.session.commit()
        return question.id

    def insert_questions(self, questions):
        self.session.add_all([Question(
            category=q['category'],
            air_date=q['air_date'],
            question=q['question'],
            value=q['value'],
            answer=q['answer'],
            round=q['round'],
            show_number=q['show_number']
        ) for q in questions])
        self.session.commit()

    def load_file(model_file: str):
        """Load JSONL from a file."""
        lines = []
        with open(model_file, "r") as model_file:
            for line in model_file:
                if line:
                    lines.append(json.loads(line))
        return lines

    def insert_questions_file(self, file_path):
        questions_data = self.load_file(file_path)
        questions = [Question(
            category=q['category'],
            air_date=q['air_date'],
            question=q['question'],
            value=q['value'],
            answer=q['answer'],
            round=q['round'],
            show_number=q['show_number']
        ) for q in questions_data]
        self.session.add_all(questions)
        self.session.commit()

    def insert_llm_response(self, llm_response: LLMResponse):
        self.session.add(llm_response)
        self.session.commit()
        return llm_response.id

    def insert_llm_judge_rating(self, llm_judge_rating: LLMJudgeRating):
        self.session.add(llm_judge_rating)
        self.session.commit()

    def get_questions(self):
        return self.session.query(Question).all()

    def insert_test_run(self, user_prompt: str, system_prompt: str, parameters: str = None):
        test_run = TestRun(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            parameters=parameters
        )
        self.session.add(test_run)
        self.session.commit()
        return test_run.id
    
    def get_test_runs(self):
        return self.session.query(TestRun).all()
    
    def get_last_test_run_id(self):
        last_test_run = self.session.query(TestRun.id).order_by(TestRun.id.desc()).first()
        if last_test_run is None:
            raise ValueError("No test runs found in the database.")
        return last_test_run[0] 

    # method to find llm responses that are not rated by the judge llm yet
    def get_unrated_llm_responses(self, llm_id: int, test_run_id: int, judge_llm: str):
        """
        Get the LLM responses that are not yet rated in a specific test run by a specific LLM and judge.
        
        Args:
            llm_id (int): The ID of the LLM.
            test_run_id (int): The ID of the test run.
            judge_llm (str): The name of the judge LLM.
        
        Returns:
            List[LLMResponse]: A list of LLMResponse objects that are not yet rated by the given judge LLM.
        """
        # Get all the LLM responses for the given LLM and test run
        llm_responses = self.get_llm_responses_by_llm_id_and_test_run_id(llm_id, test_run_id)
        
        # Get the IDs of the LLM responses that have already been rated by the given judge LLM
        rated_response_ids = self.session.query(LLMJudgeRating.llm_response_id) \
                            .join(LLMResponse, LLMResponse.id == LLMJudgeRating.llm_response_id) \
                            .filter(LLMResponse.llm_id == llm_id, LLMJudgeRating.judge_model == judge_llm, LLMJudgeRating.test_run_id == test_run_id) \
                            .distinct() \
                            .all()
        rated_response_ids = [row[0] for row in rated_response_ids]
        
        # Filter the LLM responses to get the ones that are not yet rated by the given judge LLM
        unrated_responses = [resp for resp in llm_responses if resp.id not in rated_response_ids]
        
        return unrated_responses

    def get_llm_responses_by_llm_id(self, llm_id):
        return self.session.query(LLMResponse).filter_by(llm_id=llm_id).all()

    def get_llm_judge_ratings(self) -> list[LLMJudgeRating]:
        return self.session.query(LLMJudgeRating).all()

    def get_llm_judge_models(self) -> list[LLMJudgeRating]:
        return self.session.query(LLMJudgeRating.judge_model).distinct().all()
    
    def get_judge_ratings_for_llm_by_model(self, llm_id, judge_model):
        # Fetch judge ratings for all responses of a specific LLM by a particular judge model
        return self.session.query(LLMJudgeRating).join(LLMResponse, LLMResponse.id == LLMJudgeRating.llm_response_id) \
            .filter(LLMResponse.llm_id == llm_id, LLMJudgeRating.judge_model == judge_model).all()

    # get judged ratings based on llm_response_id from LLMJudgeRating and LLM
    def get_llm_judge_ratings_by_response_and_llm(self, llm_response_id, llm_id):
        return self.session.query(LLMJudgeRating).filter_by(llm_response_id=llm_response_id, llm_id=llm_id).all()
    
    def get_evaluation_data(self):
        query = text("""
            SELECT
                llms.id,
                llms.name,
                questions.id,
                questions.question,
                questions.answer,
                llm_responses.id AS llm_response_id,
                llm_responses.response,
                llm_responses.generated_tokens,
                llm_responses.input_token_count,
                llm_judge_ratings.accuracy,
                llm_judge_ratings.coherence,
                llm_judge_ratings.completion,
                llm_judge_ratings.question_structure,
                llm_judge_ratings.generated_tokens AS judge_generated_tokens,
                llm_judge_ratings.input_token_count AS judge_input_token_count
            FROM llms
            JOIN llm_responses ON llms.id = llm_responses.llm_id
            JOIN questions ON llm_responses.question_id = questions.id
            JOIN llm_judge_ratings ON llm_responses.id = llm_judge_ratings.llm_response_id
        """)
        result = self.session.execute(query).fetchall()
        return [dict(zip(
            ['llm_id', 'llm_name', 'question_id', 'question', 'answer', 'llm_response_id', 'response', 'generated_tokens', 'input_token_count',
            'accuracy', 'coherence', 'completion', 'question_structure', 'judge_generated_tokens', 'judge_input_token_count'],
            row)) for row in result]

    def get_judgement_by_llm_response_id(self, llm_response_id):
        return self.session.query(LLMJudgeRating).filter_by(llm_response_id=llm_response_id).all()
        
    def get_llm_judge_ratings_by_response_and_judge(self, llm_response_id, judge_model):
        return self.session.query(LLMJudgeRating).filter_by(llm_response_id=llm_response_id, judge_model=judge_model).all()

    def get_llm_responses_by_llm_id_and_test_run_id(self, llm_id: int, test_run_id: int):
        return self.session.query(LLMResponse).filter_by(llm_id=llm_id, test_run_id=test_run_id).all()
    
    def get_llm_responses(self, test_run_id):
        return self.session.query(LLMResponse).filter_by(test_run_id=test_run_id).all()

    def close(self):
        self.session.close()

if __name__ == '__main__':
    # Insert data
    # questions = load_file("data/questions.jsonl")

    db = JeopardyDB(db_file='output/jeopardy.db')
    # question_id = db.insert_questions(questions)

    # # Insert test run
    # user_prompt = "You are playing the game of Jeopardy. You will be given a category and a statement. Using the information provided, you must respond with a question."
    # system_prompt = "The current date is Wednesday, April 10, 2024. The assistant is Claude, created by Anthropic. Claude's knowledge base was last updated in August 2023 and it answers user questions about events before August 2023 and after August 2023 the same way a highly informed individual from August 2023 would if they were talking to someone from Wednesday, April 10, 2024."
    # test_run_id = db.insert_test_run(user_prompt, system_prompt)
    test_run_id = db.get_last_test_run_id()
    # # Insert LLM response
    # llm_name = "GPT-3"
    # llm_response = "Who was Galileo?"
    # generated_tokens = 10
    # input_token_count = 15
    # llm_response_id = db.insert_llm_response(question_id, test_run_id, llm_name, llm_response, generated_tokens, input_token_count)

    # # Insert LLM judge rating
    # accuracy = 0.3
    # coherence = 0.7
    # completion = 0.6
    # question_structure = 0.8
    # db.insert_llm_judge_rating(llm_response_id, accuracy, coherence, completion, question_structure)

    # Retrieve data
    questions = db.get_questions()
    # test_runs = db.get_test_runs()
    llm_responses = db.get_llm_responses(test_run_id)
    #llm_judge_ratings = db.get_llm_judge_ratings(llm_response_id)

    db.close()