# AI-Twin

AI-Twin is a local Retrieval-Augmented Generation (RAG) based assistant that helps candidates respond to recruiters and job applications using their resume and LinkedIn data. It builds structured databases from a user’s profile JSON and leverages LLMs for generating context-aware replies.

---

## Features

* **Database Creation**: Converts JSON resumes into Chroma vector databases (RAG1 and RAG2 modes).
* **RAG1 Mode**: Retrieves structured context chunks (education, experience, organizations, certifications, languages).
* **RAG2 Mode**: Retrieves context based on skills and cross-links them to resume details.
* **LLM Interaction**: Uses locally hosted GGUF models via `llama-cpp-python`.
* **Chat Memory**: Maintains a rolling context window for natural conversations.
* **JD Detection**: Automatically detects if an input contains a Job Description and adjusts prompts accordingly.
* **Model Download Script**: Large models are not stored in the repo. Instead, use `download_models.py` to fetch them automatically into the `models/` folder.

---

## Project Structure

```
AI-Twin/
├── main.py                # Entry point for the interactive assistant
├── create_dbs.py          # Script to parse JSON resumes and build databases
├── define/
│   ├── functions.py       # Core utility functions
│   ├── prompts.py         # Prompt templates
├── models/
│   ├── models.py          # Contains the names of models
│   ├── download_models.py # Contains program to download gguf models
├── databases/             # Chroma vector databases + JSON profiles
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/AI-Twin.git
   cd AI-Twin
   ```

2. Set up a Python environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Linux/Mac
   venv\Scripts\activate       # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download GGUF models:

   Navigate to the `models/` directory and run:

   ```bash
   python download_models.py
   ```

   This will fetch and store all required models automatically.

---

## Usage

### Step 1: Create Databases

Run `create_dbs.py` to parse a JSON resume and generate RAG databases:

```bash
python create_dbs.py
```

* Input the JSON file path when prompted.
* The script will:

  * Save your JSON in `databases/jsons/`
  * Create vector stores in `databases/rag1_dbs/` and `databases/rag2_dbs/`
  * Assign a unique index number for the profile.

---

### Step 2: Run the Assistant

Start the interactive assistant:

```bash
python main.py
```

You will be prompted to enter:

* Index number (from DB creation step)
* RAG type (1 or 2)
* LLM model type (1–4)

Then, enter recruiter/job messages, and the assistant will generate candidate-style replies.

---

## Example Flow

1. Create DBs from `resume.json`:

   ```
   Enter json file path: resume.json
   Your Index no is: 3
   ```

2. Run `main.py`:

   ```
   Enter User's Index number: 3
   Enter RAG type (1 or 2): 1
   Enter LLM Model type (1 , 2, 3 or 4): 1
   ```

3. Chat:

   ```
   Enter your prompt: We are hiring a Data Scientist skilled in Python and ML.
   Response: Hi, I'm excited about the opportunity...
   ```

---

## Dependencies

* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) – run GGUF models locally
* [LangChain](https://www.langchain.com/) – orchestration & RAG utilities
* [ChromaDB](https://www.trychroma.com/) – vector database
* [Sentence-Transformers](https://www.sbert.net/) – embedding models
* [Pandas](https://pandas.pydata.org/) – data handling
* [tqdm](https://tqdm.github.io/) – progress bars for model downloads

---

## Notes

* This project runs **entirely locally** (no external API calls).
* You can extend it to support multiple RAG strategies, multi-profile handling, or cloud vector DBs.
* Current implementation assumes well-structured JSON resumes.

---

## License

MIT License

## Notes

### JSON File Structure

Your resume and LinkedIn data should be structured in the following JSON format:

```json
{
  "about": "",
  "Education": [
    {
      "institution": "",
      "degree": "",
      "field": "",
      "start": "",
      "end": "",
      "skills": []
    }
  ],
  "Work experience": [
    {
      "Title": "",
      "Company": "",
      "Type": "",
      "start": "",
      "end": "",
      "Location": "",
      "Responsibilities": "",
      "skills": []
    }
  ],
  "Projects": [
    {
      "Title": "",
      "Duration": "",
      "Description": "",
      "Project Link": "",
      "skills": []
    }
  ],
  "Certification": [
    {
      "Title": "",
      "Issuer": "",
      "date issued": "",
      "skills": []
    }
  ],
  "Organizations": [
    {
      "Organization": "",
      "Position": "",
      "Responsibilities": "",
      "start": "",
      "end": "",
      "skills": []
    }
  ],
  "Languages": [
    {
      "Language": "",
      "Reading proficiency": "",
      "Writing Proficiency": "",
      "Speaking proficiency": ""
    }
  ],
  "skills": []
}
```
