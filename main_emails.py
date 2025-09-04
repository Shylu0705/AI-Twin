import os

from collections import deque

from typer import prompt

from define import functions
from models import models

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

service = functions.authenticate()

email_id = functions.get_logged_in_email(service)

index = functions.get_index(email_id)

if index is None:
    print("User not found.")
    # Develop this code to run create_dbs.py as a function

rag_type = int(input("Enter RAG type (1 or 2): ").strip())

while rag_type not in [1, 2]:
    rag_type = int(input("Invalid Option. Enter RAG type (1 or 2): ").strip())

LLM_type = int(input("Enter LLM Model type (1 , 2, 3 or 4): ").strip())

while LLM_type not in [1, 2, 3, 4]:
    LLM_type = int(input("Invalid Option. Enter LLM Model type (1 , 2, 3 or 4): ").strip())

model_path = os.path.join(script_dir, "models", models.model_names[LLM_type - 1])
llm = functions.load_llm(model_path)

jsonpath, CHROMA_PATH = functions.get_user_paths(script_dir, index, rag_type)
jsondata = functions.load_json(jsonpath)

db = functions.load_db(CHROMA_PATH)

unread_messages = functions.get_unread_messages(service)

for message in unread_messages:
    if functions.check_if_JD(llm, message['body']):
        Got_JD = True
        prompt = functions.rag_email_prompt(jsondata, db, message, rag_type)

        response = llm(prompt, max_tokens=4096, stop=[])["choices"][0]["text"]

        functions.reply_to_message(service, message['id'], response)