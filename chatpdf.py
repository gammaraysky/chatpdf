"""PDF Q&A chat using 2 LLMs:

After ingesting PDF files, multi-qa-mpnet-base-dot-v1 model
is used to generate embeddings on the tokenized sentences.

When user submits a chat query, the nearest 4 matches are retrieved,
and the query and document matches are sent to gpt3.5-turbo along
with a prompt message to only reply if the document matches provide
sufficient information for it to reply to the user's query

Implemented in a gradio frontend. Usage:

Windows:
`python chatpdf.py`

Mac/Linux:
`gradio chatpdf.py`


Additional notes:
- remember to set your openai api key in a .env file,
- you can choose a default pdf file to load at launch


"""

import logging
import os

import gradio as gr
import openai
from dotenv import find_dotenv, load_dotenv

from src.documents import Documents

# from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

# * INIT

default_pdf_file_path = r"C:\Users\Han\AIAP\Career advice in the creative field.pdf"

# get openai api key
_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ["OPENAI_API_KEY"]


def main():
    with gr.Blocks(title="") as gradio_app:
        with gr.Tab(label="Document Q&A"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    chat_history = gr.Text(value=" ", lines=10, label="chat history")

                    user_query = gr.Text(label="input")
                    btn_submit = gr.Button("Chat")

                with gr.Column(scale=1):
                    # * ingest documents and generate embeddings for similarity search
                    docs = Documents([default_pdf_file_path])
                    file_output = gr.File()
                    # opt = gr.Label()
                    upload_button = gr.UploadButton(
                        "Click to Upload a File",
                        file_types=["pdf"],
                        file_count="multiple",
                    )
                    # upload_button.upload(upload_file, upload_button, file_output)
                    upload_button.upload(
                        fn=docs.upload_files,
                        inputs=upload_button,
                        outputs=[file_output],
                    )

                btn_submit.click(
                    docs.process_query,
                    inputs=[user_query, chat_history],
                    outputs=[chat_history],
                )

    gradio_app.launch(server_name="0.0.0.0", server_port=8501, debug=True)
    print("Server closed")


if __name__ == "__main__":
    main()
