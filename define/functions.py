from llama_cpp import Llama
import json
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import pandas as pd

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64
from email.mime.text import MIMEText

from . import prompts

"""
This file contains important functions required to run the LLM and create or retrieve the database along with email functions
"""
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def retrieve_relevant_chunks_rag1(db, query, k=10, score_threshold=0.5): # Main functionality of rag1
    results = db.similarity_search_with_relevance_scores(query, k=k)
    filtered_results = [(doc, score) for doc, score in results if score <= score_threshold]
    text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
    return text

def retrieve_relevant_chunks_rag2(db, jsondata, query, k=10, score_threshold=0.5): # Main functionality of rag2

    results = db.similarity_search(query, k=k)

    # For each retrieved skill, find matching full entries in jsondata
    matched_entries = []
    for doc in results:
        skill = doc.page_content.strip()

        for edu in jsondata.get("Education", []): # Education
            if skill in edu.get("skills", []):
                matched_entries.append(f"[Education]\nInstitution: {edu['institution']}\nDegree: {edu['degree']} in {edu['field']}\nDuration: {edu['start']} - {edu['end']}\nSkills: {', '.join(edu['skills'])}")

        for exp in jsondata.get("Work experience", []): # Work Experience
            if skill in exp.get("skills", []):
                matched_entries.append(f"[Work Experience]\nTitle: {exp['Title']} at {exp['Company']} ({exp['Type']})\nLocation: {exp['Location']}\nDuration: {exp['start']} - {exp['end']}\nResponsibilities:\n- " + "\n- ".join(exp['Responsibilities']) + f"\nSkills: {', '.join(exp['skills'])}")

        for org in jsondata.get("Organizations", []): # Organizations
            if skill in org.get("skills", []):
                matched_entries.append(f"[Organization]\nOrganization: {org['Organization']}\nPosition: {org['Position']}\nDuration: {org['start']} - {org['end']}\nLocation: {org['Location']}\nResponsibilities:\n- " + "\n- ".join(org['Responsibilities']) + f"\nSkills: {', '.join(org['skills'])}")

        for cert in jsondata.get("Certification", []): # Certifications
            if skill in cert.get("skills", []):
                matched_entries.append(f"[Certification]\nTitle: {cert['Title']}\nIssuer: {cert['Issuer']}\nDate Issued: {cert['date issued']}\nSkills: {', '.join(cert['skills'])}")

        for proj in jsondata.get("Projects", []): # Projects
            if skill in proj.get("skills", []):
                matched_entries.append(f"[Projects]\nTitle: {proj['Title']}\nDuration: {proj['Duration']}\nDescription: {proj['Description']}\nProject Link: {proj['Project Link']}\nSkills: {', '.join(proj['skills'])}")

        for lang in jsondata.get("Languages", []): # Languages
            if skill in lang.get("Language"):
                matched_entries.append(f"[Language]\nLanguage: {lang['Language']}\nProficiency (R/W/S): {lang['Reading proficiency']} / {lang['Writing Proficiency']} / {lang['Speaking proficiency']}")


    # Remove duplicates
    matched_entries = list(dict.fromkeys(matched_entries))

    # Join into final context
    context_text = "\n\n---\n\n".join(matched_entries)

    return context_text

def rag1_prompt(jsondata, db, query_text): # Prompt for rag1

    name = jsondata.get("Name", "Unknown")
    prefname = jsondata.get("Preferred name", "Unknown")
    address = jsondata.get("Address", "Unknown")
    about = jsondata.get("About", "Unknown")

    context_text = retrieve_relevant_chunks_rag1(db, query_text)

    prompt = ChatPromptTemplate.from_template(prompts.RAG_PROMPT_TEMPLATE).format(
        context=context_text, 
        question=query_text,
        name = name,
        address = address,
        about = about
    )

    return prompt

def rag2_prompt(jsondata, db, query_text): # Prompt for rag2

    name = jsondata.get("Name", "Unknown")
    prefname = jsondata.get("Preferred name", "Unknown")
    address = jsondata.get("Address", "Unknown")
    about = jsondata.get("About", "Unknown")

    context_text = retrieve_relevant_chunks_rag2(db, jsondata, query_text)

    prompt = ChatPromptTemplate.from_template(prompts.RAG_PROMPT_TEMPLATE).format(
        context=context_text, 
        question=query_text,
        name = name,
        address = address,
        about = about
    )

    return prompt

def rag_prompt(jsondata, db, query_text, rag_type): # Unified prompt function for both RAG1 and RAG2
    if rag_type == 1:
        return rag1_prompt(jsondata, db, query_text)
    elif rag_type == 2:
        return rag2_prompt(jsondata, db, query_text)
    else:
        raise ValueError("Invalid RAG type")

def rag1_email_prompt(jsondata, db, query_text): # Prompt for rag1 for Email

    name = jsondata.get("Name", "Unknown")
    prefname = jsondata.get("Preferred name", "Unknown")
    address = jsondata.get("Address", "Unknown")
    about = jsondata.get("About", "Unknown")

    context_text = retrieve_relevant_chunks_rag1(db, query_text)

    prompt = ChatPromptTemplate.from_template(prompts.RAG_EMAIL_PROMPT_TEMPLATE).format(
        context=context_text, 
        question=query_text,
        name = name,
        address = address,
        about = about
    )

    return prompt

def rag2_email_prompt(jsondata, db, query_text): # Prompt for rag2 for Email

    name = jsondata.get("Name", "Unknown")
    prefname = jsondata.get("Preferred name", "Unknown")
    address = jsondata.get("Address", "Unknown")
    about = jsondata.get("About", "Unknown")

    context_text = retrieve_relevant_chunks_rag2(db, jsondata, query_text)

    prompt = ChatPromptTemplate.from_template(prompts.RAG_EMAIL_PROMPT_TEMPLATE).format(
        context=context_text, 
        question=query_text,
        name = name,
        address = address,
        about = about
    )

    return prompt

def rag_email_prompt(jsondata, db, query_text, rag_type): # Unified prompt function for both RAG1 and RAG2 for Email
    if rag_type == 1:
        return rag1_prompt(jsondata, db, query_text)
    elif rag_type == 2:
        return rag2_prompt(jsondata, db, query_text)
    else:
        raise ValueError("Invalid RAG type")

def check_if_JD(llm, context_text): # Check if query contains JD

    prompt = ChatPromptTemplate.from_template(prompts.CHECK_FOR_JD).format(
        query_text=context_text
    )

    response = llm(prompt, max_tokens=4096, stop=[])["choices"][0]["text"]

    if response == "YES":
        return True
    
    return False

def load_json(jsonpath): # Load the json data shown by jsonpath
    with open(jsonpath, "r", encoding="utf-8") as f:
        jsondata = json.load(f)
    
    return jsondata

def load_db(CHROMA_PATH): # Load the chroma database
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    return db

def load_llm(llmpath): # Load the LLM model shown by llmpath
    llm = Llama(model_path=llmpath, n_ctx=4096, use_gpu=True, n_gpu_layers=-1)

    return llm

def get_user_paths(script_dir, index, rag_type): # Get Json and chromadb paths
    jsonpath = os.path.join(script_dir, "databases", "jsons", f"index_{index}.json")

    if rag_type == 1:
        chroma_path = os.path.join(script_dir, "databases", "rag1_dbs", f"index_{index}")
    elif rag_type == 2:
        chroma_path = os.path.join(script_dir, "databases", "rag2_dbs", f"index_{index}")
    else:
        return None
    
    return jsonpath, chroma_path

def add_entry(name, email): # Add entry to the CSV file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_file = os.path.join(project_root, 'databases', 'persons.csv')

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["name", "email", "index"])

    if not df.empty:
        next_index = df["index"].max() + 1
    else:
        next_index = 1

    new_row = {"name": name, "email": email, "index": next_index}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_file, index=False)
    return next_index

def rag1_chunking(data): # Generate chunks for RAG1
    chunks = []

    for edu in data["Education"]:
        text = (
            f"[Education]\n"
            f"Institution: {edu['institution']}\n"
            f"Degree: {edu['degree']} in {edu['field']}\n"
            f"Duration: {edu['start']} - {edu['end']}\n"
            f"Skills: {', '.join(edu['skills'])}"
        )
        chunks.append({"content": text, "metadata": {"type": "education", "institution": edu['institution']}})

    for exp in data["Work experience"]:
        text = (
            f"[Work Experience]\n"
            f"Title: {exp['Title']} at {exp['Company']} ({exp['Type']})\n"
            f"Location: {exp['Location']}\n"
            f"Duration: {exp['start']} - {exp['end']}\n"
            f"Responsibilities:\n- " + "\n- ".join(exp['Responsibilities']) + "\n"
            f"Skills: {', '.join(exp['skills'])}"
        )
        chunks.append({"content": text, "metadata": {"type": "experience", "title": exp['Title']}})

    for org in data["Organizations"]:
        text = (
            f"[Organization]\n"
            f"Organization: {org['Organization']}\n"
            f"Position: {org['Position']}\n"
            f"Duration: {org['start']} - {org['end']}\n"
            f"Location: {org['Location']}\n"
            f"Responsibilities:\n- " + "\n- ".join(org['Responsibilities']) + "\n"
            f"Skills: {', '.join(org['skills'])}"
        )
        chunks.append({"content": text, "metadata": {"type": "organization", "position": org['Position']}})


    for cert in data["Certification"]:
        text = (
            f"[Certification]\n"
            f"Title: {cert['Title']}\n"
            f"Issuer: {cert['Issuer']}\n"
            f"Date Issued: {cert['date issued']}\n"
            f"Skills: {', '.join(cert['skills'])}"
        )
        chunks.append({"content": text, "metadata": {"type": "certification", "title": cert['Title']}})

    for lang in data["Languages"]:
        text = (
            f"[Language]\n"
            f"Language: {lang['Language']}\n"
            f"Proficiency (R/W/S): {lang['Reading proficiency']} / "
            f"{lang['Writing Proficiency']} / {lang['Speaking proficiency']}"
        )
        chunks.append({"content": text, "metadata": {"type": "language", "language": lang['Language']}})

    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    return {"Chunks": chunks, "Texts": texts, "Metadatas": metadatas}

def rag2_chunking(data): # Generate chunks for RAG2
    skills_set = set()

    for edu in data.get("Education", []):
        skills_set.update(edu.get("skills", []))

    for exp in data.get("Work experience", []):
        skills_set.update(exp.get("skills", []))

    for org in data.get("Organizations", []):
        skills_set.update(org.get("skills", []))

    for cert in data.get("Certification", []):
        skills_set.update(cert.get("skills", []))

    for proj in data.get("Projects", []):
        skills_set.update(proj.get("skills", []))

    for lang in data.get("Languages", []):
        skills_set.add(lang.get("Language"))

    chunks = [{"content": skill.strip(), "metadata": {"type": "skill"}} for skill in skills_set if skill.strip()]

    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    return {"Chunks": chunks, "Texts": texts, "Metadatas": metadatas}

def get_path(index, name, script_dir): # Get the path of chroma dbs of the person based on the index
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_file = os.path.join(project_root, 'databases', 'persons.csv')
    df = pd.read_csv(csv_file)

    for _, i in df:
        if i["index"] == index and i["name"] == name:
            paths = [os.path.join(script_dir, "databases", "rag1_dbs", f"index_{index}")]
            return paths
    return None

def context_built_prompt(query_text, chat_history): # Create prompt with chat history if query does not contain JD
    prompt = f"Continue this conversation (each individual reply is separated by __________) (First prompt is by user, then the next is LLM, then user, and then LLM and so on):\n\n'''"

    for chat in chat_history: 
        prompt += chat + f"\n\n__________\n\n"
    
    prompt += query_text + f"\n\n'''"
    return prompt

def get_index(email): # Get the index of a user by their email
    csv_path = os.path.join(project_root, 'databases', 'persons.csv')
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        if row["email"] == email:
            return row["index"]
    return None

### Email Functions

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def authenticate(): # Authenticate user with OAuth2 and return Gmail API service.
    flow = InstalledAppFlow.from_client_secrets_file(os.path.join(project_root, "auth.json"), SCOPES)
    creds = flow.run_local_server(port=0)
    return build("gmail", "v1", credentials=creds)

def get_unread_messages(service, max_results=10):  # Fetch unread messages from Gmail inbox.
    results = service.users().messages().list(
        userId="me", labelIds=["UNREAD"], maxResults=max_results
    ).execute()

    messages = results.get("messages", [])
    full_messages = []

    for msg in messages:
        msg_id = msg["id"]
        msg_detail = service.users().messages().get(
            userId="me",
            id=msg_id,
            format="full"  # get all parts of the message
        ).execute()

        payload = msg_detail.get("payload", {})
        parts = payload.get("parts", [])

        body = ""
        if parts:
            for part in parts:
                if part.get("mimeType") == "text/plain":
                    data = part["body"].get("data", "")
                    body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                    break
        else:
            data = payload.get("body", {}).get("data", "")
            body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

        full_messages.append({"id": msg_id, "body": body})

    return full_messages

def create_reply(to, subject, body, thread_id): # Create reply message, preserving threading.
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = "Re: " + subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {"raw": raw, "threadId": thread_id}

def reply_to_message(service, msg_id, reply_text): # Reply to a specific message and mark it as read.
    message = service.users().messages().get(userId="me", id=msg_id).execute()
    headers = message["payload"]["headers"]

    sender = None
    subject = "No Subject"
    for h in headers:
        if h["name"] == "From":
            sender = h["value"]
        if h["name"] == "Subject":
            subject = h["value"]

    if sender:
        reply = create_reply(sender, subject, reply_text, message["threadId"])
        sent = service.users().messages().send(userId="me", body=reply).execute()
        print(f"Replied to {sender} with message ID: {sent['id']}")

        # Mark message as read
        mark_as_read(service, msg_id)

def mark_as_read(service, msg_id): # Mark a message as read.
    service.users().messages().modify(
        userId="me", id=msg_id, body={"removeLabelIds": ["UNREAD"]}
    ).execute()

def get_logged_in_email(service): # Return the email address of the logged in user.
    profile = service.users().getProfile(userId="me").execute()
    return profile["emailAddress"]