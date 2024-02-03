from config import api_key
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.agents.agent_types import AgentType
from utils import tiktoken
from data_loader import load_and_preprocess_data



def get_user_input():
        txt_file_path = input("Enter the path to the text file you want to process: ")
        return txt_file_path
def main():
    
    txt_file_path = "pbx.txt"
    data, vectorstore, memory = load_and_preprocess_data(txt_file_path)

    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4", max_tokens=500)
    
   
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    # Example query
    query = "what is the text about?"
    result = conversation_chain({"question": query})
    answer = result["answer"]
    print(answer)

if __name__ == "__main__":
    main()
