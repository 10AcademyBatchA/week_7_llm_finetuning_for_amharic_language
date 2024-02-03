from fastapi import FastAPI, HTTPException, Depends
from typing import Annotated, List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from utils import simple_rag
from utils import hugging_face_hub

app = FastAPI()

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)


class RagResponseBase(BaseModel):
    question: str
    answer: str


class HugResponseBase(BaseModel):
    question: str
    answer: str


class AmharicModelWithRAGBase(BaseModel):
    question: str
    answer: str


@app.get("/getanswer", response_model=RagResponseBase)
async def return_answer(question: str):
    result = simple_rag.test_RAG(question)
    return result


@app.get("/getHuggingFaceAnswer", response_model=HugResponseBase)
async def return_answer(model: str, prompt: str):
    result = hugging_face_hub.invoke_current_hugging_face_model(model, prompt)
    return result


@app.get("/getAmharicModelWithRAGAnswer", response_model=AmharicModelWithRAGBase)
async def return_answer(model: str, prompt: str):
    result = hugging_face_hub.use_amharic_model(model, prompt)
    return result
