from fastapi import FastAPI, HTTPException, Depends
from typing import Annotated, List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from utils import simple_rag

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)


class RagResponseBase(BaseModel):
    question: str
    answer: str


@app.get("/getanswer", response_model=RagResponseBase)
async def return_answer(question: str):
    result = simple_rag.test_RAG(question)
    return result
