import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

# Constants and configuration parameters
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
NEW_MODEL = "llama-2-13b-cnli-deferral"

LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT = False

OUTPUT_DIR = "./results"
NUM_TRAIN_EPOCHS = 1
FP16 = False
BF16 = False
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CHECKPOINTING = True
MAX_GRAD_NORM = 0.3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.001
OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"
MAX_STEPS = -1
WARMUP_RATIO = 0.03
GROUP_BY_LENGTH = True
SAVE_STEPS = 0
LOGGING_STEPS = 25
MAX_SEQ_LENGTH = 3800
PACKING = False
DEVICE_MAP = {"": 0}

def load_datasets(idd):
    data_files = {
        "train": f"data/balanced_single{idd}train.json",
        "test": f"data/single{idd}test.json",
        "eval": f"data/single{idd}dev.json"
    }
    return load_dataset('json', data_files=data_files)

def configure_bnb():
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    return BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )

def load_model_and_tokenizer():
    bnb_config = configure_bnb()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=DEVICE_MAP
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    
    return model, tokenizer

def load_peft_config():
    return LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
    )

def set_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIM,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        bf16=BF16,
        max_grad_norm=MAX_GRAD_NORM,
        max_steps=MAX_STEPS,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=GROUP_BY_LENGTH,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="tensorboard"
    )

def main():
    idd = str(sys.argv[1])
    dataset = load_datasets(idd)
    model, tokenizer = load_model_and_tokenizer()
    peft_config = load_peft_config()
    training_arguments = set_training_args()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=PACKING,
    )

    trainer.train()
    trainer.model.save_pretrained(f"{NEW_MODEL}{idd}")
    save_dir = f'./models_new/{idd}/{MODEL_NAME}'
    model.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
