PROMPTS = {
    "accuracy": {
        "user": "Prompt sent to the AI Assistant: {}\\n Answer from the AI Assistant: {}\\n Your rating: ",
        "system": "Please act as an impartial judge and evaluate the accuracy of the response provided by an AI assistant to the user for playing the game of Jeopardy. Respond only with a rating between 0 and 1."
    },
    "coherence": {
        "user": "Please act as an impartial judge and determine if the response provided by an AI assistant to the user was clear or understandable. Response from the AI Assistant: {}",
        "system": "Evaluate the coherence of the response provided by an AI assistant. Respond only with a rating between 0 and 1."
    },
    "completeness": {
        "user": "Evaluate the completeness of the answer '{}' to the question '{}' on a scale of 0 to 1, where 1 is highly complete and 0 is not complete at all.",
        "system": "Please act as an impartial judge and evaluate the completeness of the response provided by an AI assistant to the user for playing the game of Jeopardy. Respond only with a rating between 0 and 1."
    },
    "is_question": {
        "user": "Determine if the following text is a question (1) or not a question (0): {}",
        "system": "Please act as an impartial judge and determine if the response provided by an AI assistant to the user for playing the game of Jeopardy is a question (1) or not a question (0)."
    },
    "play": {
        "user": "You are playing the game of Jeopardy. You will be given a category and a statement. Using the information provided, you must respond with a question. Category: '{}'. Here's your statement: '{}'.",
        "system" : ""
    }
}