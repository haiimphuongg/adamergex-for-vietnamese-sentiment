import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
import json

# --- Default Configuration (can be overridden by args) ---
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
TOKENIZER_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_DROPOUT = 0.05
LORA_BIAS = "none"

# Training Arguments
PER_DEVICE_TRAIN_BATCH_SIZE = 30
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
MAX_SEQ_LENGTH = 256
OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 0.3
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 1

OUTPUT_DIR_ROOT = "./trained_adapters"

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def sentiment_formatting_func(example):
    """Formatting function for sentiment analysis dataset."""
    return f"{example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"

def clm_formatting_func(example):
    """Formatting function for causal language modeling dataset."""
    return f"{example['prompt']}{example['completion']}"

def tokenize_function(example, tokenizer, formatting_func, max_seq_length):
    """Tokenize and pad the formatted text to max_seq_length."""
    formatted_text = formatting_func(example)
    tokenized = tokenizer(
        formatted_text,
        padding="max_length",  # Pad to max_length
        truncation=True,
        max_length=max_seq_length,
        return_tensors=None,  # Return as lists, not tensors
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }

def main(args):
    print(f"\n--- Starting LoRA Adapter Training for: {args.adapter_name} ---")
    print(f"Task type: {args.task_type}")
    print(f"Using base model: {args.base_model_name}")
    print(f"Processed dataset path: {args.dataset_path}")
    adapter_specific_output_dir = os.path.join(args.output_dir_root, args.adapter_name)
    trainer_checkpoints_dir = os.path.join(adapter_specific_output_dir, "trainer_checkpoints")
    print(f"Adapter will be saved to: {adapter_specific_output_dir}")
    print(f"Trainer checkpoints: {trainer_checkpoints_dir}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load Processed Dataset
    try:
        train_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    except Exception as e:
        print(f"Error loading dataset from {args.dataset_path}: {e}")
        print("Ensure the JSON file is a list of dictionaries with appropriate fields.")
        return

    print(f"Loaded dataset with {len(train_dataset)} samples for {args.adapter_name}.")
    if len(train_dataset) == 0:
        print("Error: Dataset is empty.")
        return

    # 3. Select Formatting Function
    if args.task_type == "sentiment":
        formatting_func = sentiment_formatting_func
    elif args.task_type == "clm":
        formatting_func = clm_formatting_func
    else:
        print(f"Error: Invalid task_type '{args.task_type}'. Must be 'sentiment' or 'clm'.")
        return

    # 4. Tokenize Dataset with Padding
    tokenized_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, formatting_func, args.max_seq_length),
        batched=False,
        remove_columns=train_dataset.column_names,  # Remove original columns
    )

    # 5. Load Base Model with Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    # 6. PEFT Configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model_with_lora = get_peft_model(base_model, peft_config)
    print_trainable_parameters(model_with_lora)

    # 7. Training Arguments
    training_args = SFTConfig(
        output_dir=trainer_checkpoints_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        report_to="tensorboard",
        run_name=f"{args.adapter_name}_training_run",
        remove_unused_columns=False,  # Keep tokenized columns
        max_seq_length=args.max_seq_length,
    )

    # 8. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model_with_lora,
        train_dataset=tokenized_dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # 9. Train
    print(f"Starting SFTTrainer training for {args.adapter_name}...")
    trainer.train()
    print("Training finished.")

    # 10. Save Final Adapter
    os.makedirs(adapter_specific_output_dir, exist_ok=True)
    print(f"Saving final adapter to: {adapter_specific_output_dir}")
    trainer.model.save_pretrained(adapter_specific_output_dir)
    tokenizer.save_pretrained(adapter_specific_output_dir)
    print(f"Adapter {args.adapter_name} saved successfully.")

    # Clean up
    del model_with_lora
    del base_model
    del trainer
    torch.cuda.empty_cache()
    print(f"--- Finished training and saved: {args.adapter_name} ---\n")

if __name__ == "__main__":
    print("I am here to train LoRA adapters for Llama 3 models.")
    parser = argparse.ArgumentParser(description="Train a LoRA adapter.")
    parser.add_argument("--adapter_name", required=True, help="Name for the adapter (e.g., en_sentiment, vi_clm). This will be the subdir name.")
    parser.add_argument("--dataset_path", required=True, help="Path to the processed JSON dataset file.")
    parser.add_argument("--task_type", required=True, choices=["clm", "sentiment"], help="Task type: 'clm' for causal LM or 'sentiment' for sentiment analysis.")
    parser.add_argument("--base_model_name", type=str, default=BASE_MODEL_NAME)
    parser.add_argument("--tokenizer_name", type=str, default=TOKENIZER_NAME)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--lora_bias", type=str, default=LORA_BIAS)
    parser.add_argument("--lora_target_modules", nargs='+', default=LORA_TARGET_MODULES)
    parser.add_argument("--batch_size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--epochs", type=int, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--optim", type=str, default=OPTIM)
    parser.add_argument("--lr_scheduler_type", type=str, default=LR_SCHEDULER_TYPE)
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--logging_steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--save_total_limit", type=int, default=SAVE_TOTAL_LIMIT)
    parser.add_argument("--output_dir_root", type=str, default=OUTPUT_DIR_ROOT)

    args = parser.parse_args()
    main(args)