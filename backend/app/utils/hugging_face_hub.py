from dotenv import load_dotenv
from models import simple_rag_response

load_dotenv()

from langchain import HuggingFaceHub


def invoke_current_hugging_face_model(model, prompt):
    llm = HuggingFaceHub(
        repo_id=model, model_kwargs={"temperature": 0, "max_length": 64}
    )
    # prompt = "what are good fitness tips?"
    response = llm(prompt)
    return simple_rag_response.HugResponse(prompt, response)


def use_amharic_model(model, prompt):
    llm = HuggingFaceHub(
        repo_id=model, model_kwargs={"temperature": 0, "max_length": 64}
    )
    # prompt = "what are good fitness tips?"
    response = llm(prompt)
    return simple_rag_response.AmharicModelResponse(prompt, response)
