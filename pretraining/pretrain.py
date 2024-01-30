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
from datasets import Dataset, DatasetDict, load_dataset, load_metric, concatenate_datasets

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
            
    
    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else: batch[k] = torch.tensor([f[k] for f in features])
                
    except ValueError:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    #if v is a numpy array, then convert it to torch tensor
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else: 
                    batch[k] = torch.tensor([features[0][k]] * len(features))
                    
    return batch

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The tokenizer for weights initialization.")},
    )
    
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides cannot be used with --config_name or --model_name_or_path. To override some of "
            )
            
@dataclass
class DataTrainingArguments:
    '''
    Arguments pertaining to what data we are going to input our model for training and eval.
    '''
    
    dataset_dir: Optional[str] = field(
        default=None, metadata={"the name of the dataset to use"}
    )

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name opf the dataset to use"}
    )
    
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training file"}
    )
    
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "This is optional but recommended if you want to use early stopping"}
    )
    
    max_training_sample: Optional[int] = field(
        default=None,
        metadata={
            "help": "Debugging purposes"
        },
    )
    
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging"
        },
    )
    
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    
    # BLOCK SIZE CAN BE THE MAX INPUT FOR THE LENGTH OF THE MODEL 
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional"
                "Training dataset will be truncated into a block of this size for training"
                "Default to the model max input sequence"
            )
        }
    )
    
    # OVERWRITE CACHE DIRECTORY
    #Returns bool type
    
    cache_dir: bool = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    
    # VALIDATION STRATEGY
    #Float type
    validation_strategy: Optional[float] = field(
        default=0.01,
        metadata={
            "help": "Percentage of the validation set used at the end of each epoch"
        }
        
    )
    #PREPROCESSING NUM WORKERS
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for preprocessing"}
    )
    
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep the linebreaks when using txt files or not"}
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed store"})
    
    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets >= 2.0.0`")
            
            
@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj, v_proj")
    lora_rank : Optional[str] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.03)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    debug_mode : Optional[str] = field(default=False)
    peft_path : Optional[str] = field(default=None)
    
logger = logging.getLogger(__name__)

def main():
    
    parser = HfArgumentParser(ModelArguments, DataTrainingArguments, MyTrainingArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        #If we pass only one argument and it is a json file, lets get the arguments
        model_args, data_args, training_args = parser.parse_parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_to_dataclasses()
        
    send-example_telemetry("run_clm", model_args, data_args)
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, # if training_args.local_rank in [-1,0] else logging.WARN,
                        handlers=[logging.StreamHandler(sys.stdout)],)
    
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.warning(
        f"Process rank: {training_args.output_dir}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"Distributed training: {bool(training_args.local_rank != -1)}, 16-bits-training: {training_args.fp16}"
    )
    
    # Detecting the last checkpoint
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError (
                f"Outpur dir {training_args.output_dir} already exists and is not mt"
                "Use --overwrite_output_dir to overcome"
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this, change "
                "the --output-dir or --overwrite_output_dir to train from scratch"
            )
            
    #set the seed before initializing model
    set_seed(training_args.seed)
    
    config_kwargs = {
        "cache_dir": model.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model-args.use_auth_token else None
    }
    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else: 
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("This is a new config from scratch")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
            
            
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None
    }
    
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "Instantiating a tokenizer from scratch"
        )
        
    # Preprocessing the datasets
    #Tokenize first and then convert to features
    
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            return tokenizer(examples[text])
        
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^ PLease ignore the warning above ^^^^^^^"
            )
            
        return output

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024  
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                "Override with `--block_size xxx`"
            
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
        
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size
    
    def group_texts(examples):
        #Concatenate all texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        if total_length >= block_size:
            #We tokenize the text with small chunks of block_size
            total_length = {total_length // block_size} *  block_size
            #split by chunks of max_len
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        with training_args.main_process_first(desc="dataset map tokenizer"):
            lm_datasets = []
            path = Path(data_args.dataset_dir)
            filename = [file.name for file in path.glob("*.txt")]
            
            if training_args.debug_mode:
                files = [files[0]]
            for idx, file in enumerate(files):
                data_file = os.path.join(path, file)
                filename = ''.join(file.split('.')[:-1])
                cache_path = os.path.join(data_args.data_cache_dir, filename)
                os.makedirs(cache_path, exist_ok=True)
                try:
                    processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=True)
                    logger.info(f'Training datasets-{filename} has been loaded from disk')
                except Exception:
                    cache_dir = os.path.join(data_args.data_cache_dir, filename+"_text")
                    os.makedirs(cache_dir, exist_ok=True)
                    raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
                    logger.info(f"{file} has been loaded")
                    tokenized_dataset = raw_dataset.map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns="text",
                        load_from_cache_file=True,
                        keep_in_memory=False,
                        cache_file_names = {k: os.path.join(cache_dir, "tokenized.arrow") for k in raw_dataset},
                        desc="Running tokenizer on the dataset",
                    )
                    
                    grouped_datasets = tokenized_dataset.map(
                        group_texts,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=True,
                        keep_in_memory=False,
                        cache_file_names = {k: os.path.join(cache_dir, "grouped.arrow") for k in tokenized_dataset},
                        desc=f'Grouping texts in chunks of {block_size}',
            
                    )
                    
                    processed_dataset = grouped_datasets
                    processed_dataset.save_to_disk(cache_path)
                    
                if idx == 0:
                    lm_datasets = processed_dataset['train']
                else:
                    assert lm_datasets.features.type == processed_dataset['train'].features.type
                    lm_dataset = concatenate_datasets([lm_datasets, processed_dataset['train']])
                    
            lm_datasets = lm_datasets.train_test_split(test_size= data_args.validation_split_percentage())
            
        if training_args.do_train:
            train_dataset = lm_datasets["train"]
            
            
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
                logger.info(f"Num train samples {len(train_dataset)}")
                logger.info("Training example: ")
                logger.info(tokenizer.decode(train_dataset[0]["input_ids"]))
                
                
        if model_args.model_name_or_path:
            torch_dtype = (
                model_args.torch_dtype
                if model_args.torch_dtype in ["auto", None]
                else getattr(torch, model_args.torch_dtype)
            )
            
            model = LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".cpkt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            
        else:
            model = AutoModelForCausalLM.from_config(config)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M parameters")
            