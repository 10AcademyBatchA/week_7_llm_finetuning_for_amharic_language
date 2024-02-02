import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
from datasets import load_dataset, DatasetDict
from peft import prepare_model_for_kbit_training
from huggingface_hub import login

tokenizer = "BiniyamAjaw/amharic_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer)

from os import path
dataset_dict = load_dataset('text', data_files='/home/biniyam_ajaw/finetuning/merged/merged.txt')

dataset = dataset_dict['train']

train_test = dataset.train_test_split(test_size=0.2)
dataset = train_test


context_len = 1024
def tokenize(element):
    return tokenizer(
        element['text'],
        truncation=True,
        padding = True,
        max_length = context_len,
        return_overflowing_tokens=True
    )
    
tokenized_datasets = dataset.map(
    tokenize, batched=True, remove_columns=dataset['train'].column_names
)

login()
dataset.push_to_hub("BiniyamAjaw/amharic_dataset_v2")