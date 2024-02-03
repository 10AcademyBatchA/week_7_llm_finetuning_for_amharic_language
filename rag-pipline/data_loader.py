from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

def load_and_preprocess_data(txt_file_path):
    loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
    data = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=3100, chunk_overlap=200)
    data = text_splitter.split_documents(data)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(data, embedding=embeddings)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    return data, vectorstore, memory
