import os
import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration & Model Parameters ---
MODEL_NAME_OR_PATH = "xlm-roberta-base"
MAX_LENGTH = 128  # Max sequence length for tokenizer (consistent with dataset filtering)

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES_SENTIMENT = ["query", "value", "key", "dense"]  # For XLM-RoBERTa attention layers
LORA_TARGET_MODULES_MLM = ["query", "value", "key", "dense"]  # For MLM, include dense layers

# Label mapping for sentiment (consistent with process_data.py)
SENTIMENT_LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL_SENTIMENT = {v: k for k, v in SENTIMENT_LABEL_MAP.items()}
LABEL2ID_SENTIMENT = SENTIMENT_LABEL_MAP
NUM_LABELS_SENTIMENT = len(SENTIMENT_LABEL_MAP)

# --- Helper Functions ---
def safe_create_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def compute_metrics_sentiment(pred):
    """Compute accuracy and F1 score for sentiment classification."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# --- Main Training Function ---
def train_adapter(task, data_path, output_dir_base):
    """Trains a LoRA adapter for the specified task (mlm_en, mlm_vi, or sentiment_vi)."""
    logging.info(f"Starting training for task: {task}")
    safe_create_dir(output_dir_base)

    # Determine task type and language
    if task == "mlm_en":
        task_type = "mlm"
        lang = "en"
    elif task == "mlm_vi":
        task_type = "mlm"
        lang = "vi"
    elif task == "sentiment_vi":
        task_type = "sentiment"
        lang = "vi"
    elif task == 'sentiment_en':
        task_type = "sentiment"
        lang = "en"
    else:
        logging.error(f"Invalid task: {task}. Must be 'mlm_en', 'mlm_vi', 'sentiment_en' or 'sentiment_vi'.")
        return

    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        return

    # 2. Load Processed Dataset
    try:
        if not os.path.exists(data_path):
            logging.error(f"Processed dataset not found at {data_path}. Run process_data.py first.")
            return
        dataset = load_from_disk(data_path)
        logging.info(f"Loaded dataset from {data_path}")
    except Exception as e:
        logging.error(f"Could not load dataset from {data_path}: {e}")
        return

    # 3. Preprocess (Tokenize) Data
    def tokenize_function_sentiment(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    def tokenize_function_mlm(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    if task_type == "sentiment":
        tokenized_dataset = dataset.map(tokenize_function_sentiment, batched=True, remove_columns=["text"])
    elif task_type == "mlm":
        tokenized_dataset = dataset.map(tokenize_function_mlm, batched=True, remove_columns=["text"])
    else:
        logging.error(f"Unknown task_type: {task_type}")
        return

    # 4. Initialize Model, LoRA Config, and PEFT Model
    if task_type == "sentiment":
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME_OR_PATH,
                num_labels=NUM_LABELS_SENTIMENT,
                id2label=ID2LABEL_SENTIMENT,
                label2id=LABEL2ID_SENTIMENT,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            logging.error(f"Error loading sentiment model: {e}")
            return
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES_SENTIMENT,
            bias="none",
            modules_to_save=["classifier"]
        )
    elif task_type == "mlm":
        try:
            model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME_OR_PATH)
        except Exception as e:
            logging.error(f"Error loading MLM model: {e}")
            return
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES_MLM,
            bias="none"
        )

    try:
        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()
    except Exception as e:
        logging.error(f"Error creating PEFT model: {e}")
        return

    # 5. Data Collator
    if task_type == "sentiment":
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    elif task_type == "mlm":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # 6. Training Arguments
    adapter_name = f"{lang}_{task_type}_xlmr_lora"
    training_output_dir = os.path.join(output_dir_base, adapter_name, "training_checkpoints")
    final_adapter_save_path = os.path.join(output_dir_base, adapter_name)

    # Optimize for 8GB RAM
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        num_train_epochs=3,  # 1 epoch for MLM due to large dataset
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir_base, adapter_name, "logs"),
        logging_steps=100,
        eval_strategy="epoch" if "validation" in tokenized_dataset else "no",
        save_strategy="epoch" if "validation" in tokenized_dataset else "no",
        load_best_model_at_end="validation" in tokenized_dataset and task_type == "sentiment",  # Only for sentiment
        metric_for_best_model="f1" if task_type == "sentiment" and "validation" in tokenized_dataset else None,
        save_total_limit=1,
        fp16=False,  # Disable mixed precision for CPU/low-VRAM GPU
        report_to="none"
    )

    # 7. Trainer
    try:
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_sentiment if task_type == "sentiment" else None,
        )
    except Exception as e:
        logging.error(f"Error creating Trainer: {e}")
        return

    # 8. Train
    logging.info(f"Starting training for {adapter_name}...")
    trainer.train()
    #     logging.info(f"Training finished for {adapter_name}.")
    # except Exception as e:
    #     logging.error(f"Training failed for {adapter_name}: {e}")
    #     return

    # 9. Save Adapter
    try:
        safe_create_dir(final_adapter_save_path)
        peft_model.save_pretrained(final_adapter_save_path)
        tokenizer.save_pretrained(final_adapter_save_path)
        logging.info(f"Adapter saved to {final_adapter_save_path}")
    except Exception as e:
        logging.error(f"Error saving adapter to {final_adapter_save_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA adapters for MLM (English/Vietnamese) or Sentiment (Vietnamese).")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["mlm_en", "mlm_vi", "sentiment_vi", "sentiment_en"],
        help="Task to train: 'mlm_en' (English MLM), 'mlm_vi' (Vietnamese MLM), or 'sentiment_vi' (Vietnamese Sentiment)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the processed Hugging Face dataset directory (e.g., ./data/processed/vi_mlm/)."
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="./adapters/",
        help="Base directory to save trained adapters (default: ./adapters/)."
    )

    args = parser.parse_args()

    # Example command-line calls:
    # python train_adapter.py --task mlm_en --data_path ./data/processed/en_mlm/ --output_base ./adapters/
    # python train_adapter.py --task mlm_vi --data_path ./data/processed/vi_mlm/ --output_base ./adapters/
    # python train_adapter.py --task sentiment_en --data_path ./data/processed/en_sentiment/ --output_base ./adapters/
    # import pandas as pd
    # # Load the sentiment dataset and show the distribution of labels
    # dataset = load_from_disk(args.data_path)    
    # if "train" in dataset:
    #     train_dataset = dataset["train"]
    #     label_column = train_dataset["labels"]
    #     label_column = pd.Series(label_column)
    #     label_counts = label_column.value_counts()
    #     logging.info(f"Label distribution in training set: {label_counts.to_dict()}")
    # else:
    #     logging.warning("No training set found in the dataset.")
    

    train_adapter(args.task, args.data_path, args.output_base)