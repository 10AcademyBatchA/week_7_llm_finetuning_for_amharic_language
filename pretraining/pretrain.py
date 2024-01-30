import logging
import numpy as np
import math
import os, sys
import torch
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Tuple, Dict, Any, Mapping
from pathlib import Path
import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_metric

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,  
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    is_torch_gpu_available,
)

from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.testing_utils import CaptureLogger
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version_core

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "pt_lora-Model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            
        
        peft_model_path = os.path.join(checkpoint_folder, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)
        
    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

def accuracy(predictions, references, normalize=True, sample_weight=None):
    return {
        "accuracy": float(
            accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
        )
    }

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}
    
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["label"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
            
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
            
            


