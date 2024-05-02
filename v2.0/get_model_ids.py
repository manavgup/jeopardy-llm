from pprint import pprint
import sys

from dotenv import load_dotenv

from genai.client import Client
from genai.credentials import Credentials

# Load environment variables
load_dotenv()

# Initialize the GENAI client
client = Client(credentials=Credentials.from_env())

def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"

# Redirect output to a text file
sys.stdout = open('output.txt', 'w')

# Print and fetch model data
print(heading("List all models"))
for model in client.model.list(limit=100).results:
    print(model.model_dump(include=["name", "id"]))

for model in client.model.list(limit=100).results:
    print(heading("Get model detail"))
    model_detail = client.model.retrieve(model.id).result
    pprint(model_detail.model_dump(include=["name", "description", "id", "developer", "size"]))

# Close the file to ensure all data is written properly
sys.stdout.close()

# Reset stdout to default to allow further prints to go to the console
sys.stdout = sys.__stdout__