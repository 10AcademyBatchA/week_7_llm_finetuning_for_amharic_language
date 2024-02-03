from tokenizers import Tokenizer
from tokenizers.models import BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=80000)

import tokenizers

from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

folder = './data'
files = [f"/home/biniyam_ajaw/finetuning/{folder}/{split}.csv" for split in ["test", "train", "valid"]]
tokenizer.train(files, trainer)

from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

from transformers import PreTrainedTokenizerFast

custom_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
custom_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
custom_tokenizer.save_pretrained("amharic_tokenizer")

custom_tokenizer.push_to_hub("amharic_tokenizer")


