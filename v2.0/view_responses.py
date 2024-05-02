import gradio as gr
from db_operations import JeopardyDB  
from judge_manager import JudgeManager
import plotly.graph_objs as go
import random
from collections import defaultdict

# Initialize database
database_file = 'output/jeopardy.db'
db = JeopardyDB(db_file=database_file)

def get_llm_options():
    llms = db.get_all_llms()
    return [(llm.name, llm.id) for llm in llms]

def get_judge_options():
    judges = db.get_all_judge_llms()
    return [(judge.name) for judge in judges]

def display_llm_responses(selected_llm_id):
    # Clear the existing content
    llm_responses.value = ""

    responses = db.get_llm_responses_by_llm_id(selected_llm_id)
    if not responses:
        return "No responses found for the selected LLM."

    # Create an HTML table for the responses with an ID and extra columns for judge ratings
    table_html = ("<table id='response_table' style='width:100%; border: 1px solid black;'>"
                  "<tr><th>Prompt</th><th>Response</th><th>Generated Tokens</th><th>Input Token Count</th>"
                  "<th>Accuracy</th><th>Coherence</th><th>Completion</th><th>Question Structure</th>"
                  "<th>Judge Generated Tokens</th><th>Judge Input Tokens</th></tr>")
    
    for res in responses:
        # Add empty cells for the judge ratings
        table_html += (f"<tr><td>{res.prompt}</td><td>{res.response}</td><td>{res.generated_tokens}</td>"
                       f"<td>{res.input_token_count}</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>")
    
    table_html += "</table>"
    return table_html

def display_judge_ratings(selected_llm_id, selected_judge):
    # Clear the existing content
    llm_responses.value = ""
    judge_ratings.value = ""

    responses = db.get_llm_responses_by_llm_id(selected_llm_id)
    ratings = db.get_judge_ratings_for_llm_by_model(selected_llm_id, selected_judge)
    rating_dict = {rating.llm_response_id: rating for rating in ratings}

    table_html = ("<table id='response_table' style='width:100%; border: 1px solid black;'>"
                  "<tr><th>Prompt</th><th>Response</th><th>Generated Tokens</th><th>Input Token Count</th>"
                  "<th>Accuracy</th><th>Coherence</th><th>Completion</th><th>Question Structure</th>"
                  "<th>Judge Generated Tokens</th><th>Judge Input Tokens</th></tr>")
    
    for res in responses:
        rating = rating_dict.get(res.id, None)
        if rating:
            table_html += (f"<tr><td>{res.prompt}</td><td>{res.response}</td><td>{res.generated_tokens}</td>"
                           f"<td>{res.input_token_count}</td><td>{rating.accuracy}</td><td>{rating.coherence}</td>"
                           f"<td>{rating.completion}</td><td>{rating.question_structure}</td>"
                           f"<td>{rating.generated_tokens}</td><td>{rating.input_token_count}</td></tr>")
        else:
            # Fill with empty ratings if no ratings available
            table_html += (f"<tr><td>{res.prompt}</td><td>{res.response}</td><td>{res.generated_tokens}</td>"
                           f"<td>{res.input_token_count}</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>")
    
    table_html += "</table>"
    llm_responses.value = table_html
    judge_ratings.value = table_html

def generate_judgement(selected_llm_id: int, judge_llm: str):
    # Call the generate_judgement function
    manager.generate_judgements(database_file, judge_llm, selected_llm_id)

    # Update the LLM dropdown, Judge dropdown, and Judge ratings
    llm_responses.value = display_llm_responses(selected_llm_id)
    judge_ratings.value = display_judge_ratings(selected_llm_id, judge_llm)

    return llm_responses.value, judge_ratings.value

def plot_spider_chart():
    # Get the evaluation data from the database
    evaluation_data = db.get_evaluation_data()

    # Group the data by LLM
    llm_data = defaultdict(list)
    for data in evaluation_data:
        llm_data[data['llm_name']].append(data)

    # Create the spider chart
    metrics = ['Accuracy', 'Coherence', 'Completion', 'Question Structure', 'Avg. Generated Tokens', 'Avg. Input Tokens', 'Avg. Judge Input Tokens']
    data = []
    colors = [f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})' for _ in range(len(llm_data))]

    for i, (llm_name, llm_responses) in enumerate(llm_data.items()):
        filtered_responses = [r for r in llm_responses if None not in r.values()]
        if not filtered_responses:
            continue
        accuracy = sum(map(lambda r: r['accuracy'], filtered_responses)) / len(filtered_responses)
        coherence = sum(map(lambda r: r['coherence'], filtered_responses)) / len(filtered_responses)
        completion = sum(map(lambda r: r['completion'], filtered_responses)) / len(filtered_responses)
        question_structure = sum(map(lambda r: r['question_structure'], filtered_responses)) / len(filtered_responses)
        avg_generated_tokens = sum(map(lambda r: r['generated_tokens'], llm_responses)) / len(llm_responses)
        avg_input_tokens = sum(map(lambda r: r['input_token_count'], llm_responses)) / len(llm_responses)
        avg_judge_input_tokens = sum(map(lambda r: r['judge_input_token_count'], llm_responses)) / len(llm_responses)

        trace = go.Scatterpolar(
            r=[accuracy, coherence, completion, question_structure, avg_generated_tokens, avg_input_tokens, avg_judge_input_tokens],
            theta=metrics,
            fill='toself',
            name=llm_name,
            line=dict(color=colors[i], width=2)
        )
        data.append(trace)

    layout = go.Layout(
        title='LLM Performance Comparison',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)
    spider_chart.value = fig.to_html(full_html=False, default_height=500, default_width=800)
    return fig

with gr.Blocks() as interface:
    manager = JudgeManager(database_file)
    llm_dropdown = gr.Dropdown(label="Select LLM", choices=get_llm_options())
    judge_dropdown = gr.Dropdown(label="Select Judge", choices=get_judge_options())
    gen_judgement_button = gr.Button(value="Generate Judgement")
    plot_chart_button = gr.Button(value="Plot Spider Chart")
    llm_responses = gr.HTML()
    judge_ratings = gr.HTML()
    spider_chart = gr.HTML()

    llm_dropdown.change(fn=display_llm_responses, inputs=llm_dropdown, outputs=llm_responses)
    judge_dropdown.change(fn=display_judge_ratings, inputs=[llm_dropdown, judge_dropdown], outputs=llm_responses)
    gen_judgement_button.click(fn=generate_judgement, inputs=[llm_dropdown, judge_dropdown], outputs=[llm_responses, judge_ratings])
    plot_chart_button.click(fn=plot_spider_chart, inputs=[], outputs=[gr.Plot()])

    interface.launch()
