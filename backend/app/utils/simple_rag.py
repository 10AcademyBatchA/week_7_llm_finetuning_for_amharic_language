from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()


def load_data():
    loader = TextLoader("./week_6_challenge_doc.txt")
    documents = loader.load()
    return documents


def return_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts


def return_chain(texts):
    embeddings = OpenAIEmbeddings()
    store = Chroma.from_documents(
        texts, embeddings, collection_name="challenge_document"
    )
    llm = OpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())
