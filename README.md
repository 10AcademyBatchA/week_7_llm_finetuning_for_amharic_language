# LLM Fine-Tuning for Amharic Language

## Introduction

Welcome to the LLM Fine-Tuning for Amharic Language repository, initiated by AIQEM, an African startup specializing in AI and Blockchain solutions. Our goal is to enhance technological innovations' impact in Ethiopia and Africa's business landscape. Our latest flagship project, Adbar, is an AI-based Telegram Ad solution that optimally places ads on Telegram channels through data analysis and bots.

## Business Need

As Telegram grows, AIQEM adapts its advertising strategy to align with this platform. Our focus is on improving ad effectiveness by integrating powerful AI for Amharic text manipulation. We aim to create an Amharic RAG pipeline that generates creative text ads for Telegram channels based on campaign details, including brand and product information.

Success ensures ads are catchy and relevant to the Telegram community. To achieve this, we need quality Amharic text embedding and generation capabilities. This involves fine-tuning open-source Large Language Models (LLMs) like Mistral, Llama 2, Falcon, or Stable AI 2 to meet business objectives.

## SLMS and LLMS Finetuned in this project

| Model | Parameters |
|------------------|------------------|
| Microsoft Phi | 2B |
| StableLM  | 2B |
| LLaMA 2 | 7B |
| Mistral  | 7B |

## Tokenizer

A tokenizer with 100k vocabulairies that was trained by bpe (Byte-pair-encoding)
Trained a custom tokenizer for the language that is currently available for inference on huggingface on the following link

   `https://huggingface.co/BiniyamAjaw/amharic_tokenizer/blob/main/README.md`

## Dataset

Dataset that was gathered from public telegram channels that are in the following categories

- News
- Sports
- Literature
- E-comerece

Data preparation and preprocessing pipeline was created in order to create a huge corpus of data
The data is available on huggingface on the following link
   `https://huggingface.co/datasets/BiniyamAjaw/amharic_dataset_v2`


## Project Overview

The project is divided into tasks:

1. **Literature Review & Huggingface Ecosystem:**
   - Understand LLMs and explore Huggingface for fine-tuning.
2. **Load an LLM and Use It for Inference:**
   - Set up the environment and test model inference.
3. **Data Preprocessing and Preparation:**
   - Clean Telegram data for fine-tuning.
4. **Fine-Tuning the LLM:**
   - Train and fine-tune LLMs for Amharic text.
5. **Build RAG Pipeline for Amharic Ad Generation:**
   - Implement RAG techniques for ad content generation.

## Repository Structure

- **pretraining:** Scripts for pretraining on the corpus.
- **notebooks:** Jupyter notebooks for analysis.
- **utils:** Helper functions.
- **backend:** FastAPI backend.
- **frontend:** React frontend.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/biniyam69/Amharic-LLM-Finetuning.git
   ```

2. **Setup Backend:**
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

3. **Setup Frontend:**
   ```bash
   cd ../frontend
   npm install
   npm start
   ```

4. **Access the RAG Ad Builder:**
   - Open your browser and go to [http://localhost:3000](http://localhost:3000).

## License

This project is licensed under the MIT License.

Feel free to contribute, provide feedback, or use the code as needed!


## Part of the Project
* Back-End:[Back-End-Folder](https://github.com/10AcademyBatchA/week_7_llm_finetuning_for_amharic_language/tree/development/backend)
* Front-End:[Front-End-Folder](https://github.com/10AcademyBatchA/week_7_llm_finetuning_for_amharic_language/tree/development/front-end)

