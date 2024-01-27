# LLM Finetuning

## Enabling Quality Embedding and Text Generation for Amharic Language

AIQEM, an African startup specializing in AI and Blockchain solutions, is driven by a commitment to make a substantial impact on Ethiopian and African businesses through cutting-edge innovations in AI and Blockchain technologies. The company's latest flagship project, Adbar, is an end-to-end AI-based Telegram Ad solution recently launched in the Ethiopian market. Adbar utilizes a network of Telegram bots and sophisticated data analysis to strategically place ads across various Telegram channels.

Recognizing the increasing prominence of Telegram as a messaging platform, AIQEM acknowledges the necessity of refining its advertising strategy to harmonize with this evolving ecosystem. This report explores and summarizes key topics, including LLM, Transformer, Tokenization, embedding, Hugging Face, and more. These elements are crucial in comprehending and shaping the dynamic ecosystem, providing AIQEM with an enhanced foundation for effective ad generation.

## Part of the Project

- Back-End:[Back-End-Folder](https://github.com/10AcademyBatchA/week_7_llm_finetuning_for_amharic_language/tree/development/backend)
- Front-End:[Front-End-Folder](https://github.com/10AcademyBatchA/week_7_llm_finetuning_for_amharic_language/tree/development/front-end)

You wil find a designated folders for each implementation. Moreover, a react app is on development for facilitating the buisness requirements.

---

## Folder Structure

### You will find the following directories in this project

1. `backend/app/` - consists api endpoints for a FastAPI development
2. `front-end/screenshots` - screenshots of the frontend UI
3. `front-end/api` - axios api implementation to connect front end to backend
4. `backend/notebooks` - in this folder you will find notebooks that we used to test out different tokenizers and different trials of finetuning the model
5. `front-end/` - you will find the frontend react app in this directory.

---

## Tech Stack

We have used the following techstacks for making this Application

- React
- Langchain
- Huggingface
- FastApi

---

## Installation

Before installing this app, you need to prepare a few things

- install node.js
- install the required python packages based on the requirement file

your config file should look like this

```bash
# create huggingface api account and enter the following
HUGGINGFACEHUB_API_TOKEN = "your api key"
```

```bash
git clone https://github.com/10AcademyBatchA/week_7_llm_finetuning_for_amharic_language

cd week_7_llm_finetuning_for_amharic_language

# start api
cd backend
cd app
uvicorn app:main --reload

# start react frontend
# go to the react folder first
cd ../front-end
npm run dev

```
