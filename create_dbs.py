import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from define import functions

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Authenticate the user using Google OAuth
service = functions.authenticate()

jsonpath = input("Enter json file path: ")

# Open the json file (Didnt add error handling)
with open(jsonpath, "r", encoding="utf-8") as f:
    data = json.load(f)

name = data.get("Name", "Unknown")
email = functions.get_logged_in_email(service)

# Authenticate email ID
## To be completed

# Add entry to DataFrame
index = functions.add_entry(name, email)

print(f"Your Index no is: {index}")

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Creating DB for RAG1

rag1_output = functions.rag1_chunking(data)

path1 = os.path.join(script_dir, "databases", "rag1_dbs", f"index_{index}")

vectorstore = Chroma.from_texts(texts=rag1_output['Texts'], embedding=embedding_model, metadatas=rag1_output['Metadatas'], persist_directory=path1)  # Fixed key and updated embedding

# Creating DB for RAG2

rag2_output = functions.rag2_chunking(data)

path2 = os.path.join(script_dir, "databases", "rag2_dbs", f"index_{index}")

vectorstore = Chroma.from_texts(texts=rag2_output['Texts'], embedding=embedding_model, metadatas=rag2_output['Metadatas'], persist_directory=path2)  # Fixed key and updated embedding

# Save the json data

path3 = os.path.join(script_dir, "databases", "jsons", f"index_{index}.json")

os.makedirs(os.path.dirname(path3), exist_ok=True)

with open(path3, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
