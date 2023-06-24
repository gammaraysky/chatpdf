import glob
import logging
import os
import re

import faiss
import nltk
import torch
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pdfminer.high_level import extract_text
from transformers import AutoModel, AutoTokenizer


class Documents:
    def __init__(self, file_paths=None):
        if file_paths is not None:
            self.update_docs(file_paths)

    def update_docs(self, file_paths=None):
        self.pdf_files = file_paths

        self.sentences, self.embeddings, self.index = self.process_documents(
            self.pdf_files
        )

    def upload_files(self, files):
        file_paths = [file.name for file in files]
        if file_paths is not None and len(file_paths) > 0:
            self.update_docs(file_paths)
        # return file_paths, Documents(file_paths=file_paths)

    def extract_raw_text(self, text):
        # Remove HTML tags using BeautifulSoup
        # soup = BeautifulSoup(text, "html.parser")
        # text = soup.get_text()

        # Remove special characters and formatting using regular expressions
        # text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
        text = re.sub(r"\n", " ", text)  # Remove newline characters
        text = re.sub(r"\s+", " ", text)  # Remove extra whitespace again
        # text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters

        return text.strip()

    def generate_embeddings(self, texts):
        model = AutoModel.from_pretrained(
            "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        )
        input_ids = torch.tensor(
            tokenizer.batch_encode_plus(texts, add_special_tokens=True, padding=True)[
                "input_ids"
            ]
        )
        with torch.no_grad():
            outputs = model(input_ids)

        embeddings = outputs[0][
            :, 0, :
        ].numpy()  # Extract embeddings from the last layer

        return embeddings

    def perform_similarity_search(self, query, k=4):
        query_embedding = self.generate_embeddings([query])
        _, indices = self.index.search(query_embedding, k)
        return indices[0]

    def ingest_pdf_folder(self, folder_path: str):
        # Ingest PDF documents and tokenize sentences
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        # logging.info(pdf_files)
        return pdf_files

    def process_documents(self, pdf_files):
        sentences = []

        # Tokenize into sentences
        for pdf_file in pdf_files:
            logging.info("%s", pdf_file)
            text = extract_text(pdf_file)
            text = self.extract_raw_text(text)
            logging.info("extracted text: %s", text[:50])
            tokenized_text = nltk.sent_tokenize(text)
            sentences.extend(tokenized_text)

        # Generate embeddings
        embeddings = self.generate_embeddings(sentences)

        # Set up FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        return sentences, embeddings, index

    def process_query(self, query: str, chat_history: str) -> str:
        # * Similarity search and get top 4 sentences which match query
        results = self.perform_similarity_search(query, k=4)

        # * Expand results to include surrounding sentences to contain more information
        # ? results will store top 4 sentences which are closest match.
        # ? however, we want to give more information to the LLM
        # ? for each result get the surrounding sentences, so plus and minus 10 sentences
        paragraphs = []
        for result in results:
            paragraph = []

            if result - 5 > 0 and result + 5 < len(self.sentences):
                paragraph = " ".join(self.sentences[result - 5 : result + 5])
            elif result - 5 > 0:
                paragraph = " ".join(self.sentences[result - 5 :])
            elif result + 5 < len(self.sentences):
                paragraph = " ".join(self.sentences[: result + 5])
            paragraphs.append(paragraph)

        # * prompt LLM to review query and find the best answer from the information scraped

        # To control the randomness and creativity of the generated
        # text by an LLM, use temperature = 0.0
        chat = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

        template_string = """
        Answer the question below (delimited by triple backticks) using only document provided below (delimited by triple backticks). You do not need to use all the information, only use the most relevant.
        Question: ```{query}```
        Document: ```{paragraphs}```
        """

        prompt_template = ChatPromptTemplate.from_template(template_string)

        prompt = prompt_template.format_messages(query=query, paragraphs=paragraphs)

        # Call the LLM to translate to the style of the customer message
        answer = chat(prompt)

        result = chat_history + "\n\n" + query + "\n\n" + str(answer.content)

        return result


# def get_completion(prompt, model="gpt-3.5-turbo"):
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0,
#     )
#     return response.choices[0].message["content"]
