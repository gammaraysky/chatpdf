# ChatPDF - PDF Q&A chat using 2 LLMs:

After ingesting PDF files, `multi-qa-mpnet-base-dot-v1` model
is used to generate embeddings on the tokenized sentences.

When user submits a chat query, the nearest 4 matches are retrieved,
and the query and document matches are sent to `gpt3.5-turbo` along
with a prompt message to only reply if the document matches provide
sufficient information for it to reply to the user's query

Implemented in a gradio frontend, showing the current chat history, user
input textbox, and file upload button to upload PDF files.


---

### Setup:

1. Create a new environment.

2. Run `pip install -r requirements.txt`

3. Run: `python setup.py` (this installs some nltk data for tokenisation)

---

### Usage:

-   Windows: `python chatpdf.py`

-   Mac/Linux: `gradio chatpdf.py`

After that, open `localhost:8501` on your browser.

---

### Additional notes:

- remember to set your `OPENAI_API_KEY` in a `.env` file
- you can choose a default PDF file to load at launch