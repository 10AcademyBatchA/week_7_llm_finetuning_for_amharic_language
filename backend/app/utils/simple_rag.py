from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from models import simple_rag_response
import os
from dotenv import load_dotenv

load_dotenv()


def load_data():
    loader = TextLoader("../app/utils/week_6_challenge_doc.txt", encoding="UTF-8")
    documents = loader.load()
    return documents


def return_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)
    return texts


def return_chain(texts):
    embeddings = OpenAIEmbeddings()
    store = Chroma.from_documents(
        texts, embeddings, collection_name="challenge_document"
    )
    llm = OpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())


def test_RAG(question):
    documents = load_data()
    chunks = return_chunks(documents)
    chain = return_chain(chunks)
    response = chain.run(question)
    return simple_rag_response.RagResponse(question, response)
