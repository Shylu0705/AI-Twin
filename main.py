import os

from collections import deque

from define import functions
from models import models
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define context window and chat history FOR LLM
context_window_size = 3
chat_history = deque(maxlen=context_window_size * 2)

# Server side details (Didnt include any error handling)
index = input("Enter User's Index number: ")

rag_type = int(input("Enter RAG type (1 or 2): ").strip())

while rag_type not in [1, 2]:
    rag_type = int(input("Invalid Option. Enter RAG type (1 or 2): ").strip())

LLM_type = int(input("Enter LLM Model type (1 , 2, 3 or 4): ").strip())

while LLM_type not in [1, 2, 3, 4]:
    LLM_type = int(input("Invalid Option. Enter LLM Model type (1 , 2, 3 or 4): ").strip())

# Load the selected LLM model
model_path = os.path.join(script_dir, "models", models.model_names[LLM_type - 1])
llm = functions.load_llm(model_path)

jsonpath, CHROMA_PATH = functions.get_user_paths(script_dir, index, rag_type) 

Got_JD = False
jsondata = functions.load_json(jsonpath)
db = functions.load_db(CHROMA_PATH)

while True:
    query_text = input("Enter your prompt: ")
    Got_JD = functions.check_if_JD(llm, query_text)

    if Got_JD:
        prompt = functions.rag_prompt(jsondata, db, query_text, rag_type)

        chat_history.clear()

    else:
        prompt = functions.context_built_prompt(query_text, chat_history)
    
    response = llm(prompt, max_tokens=4096, stop=[])["choices"][0]["text"]

    chat_history.append(prompt)
    chat_history.append(response)

    print("Response:", response)
