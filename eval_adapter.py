import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import PeftModel, PeftConfig, get_peft_model
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_NAME_OR_PATH = "xlm-roberta-base"  # Base XLM-R model
MAX_LENGTH = 128  # Max sequence length for tokenizer

# Paths to the trained adapters (consistent with train_adapter.py)
ADAPTER_BASE_DIR = "./adapters/"
ADAPTER_ENG_SENTIMENT_NAME = "en_sentiment_xlmr_lora"  # A_l2t1
ADAPTER_VI_REF_NAME = "vi_mlm_xlmr_lora"              # A_l1t2 (MLM as reference)
ADAPTER_ENG_REF_NAME = "en_mlm_xlmr_lora"             # A_l2t2 (MLM as reference)

# Paths to the evaluation datasets
EVAL_DATASETS = {
    "aivivn": {
        "path": "/workspace/adamergex-for-vietnamese-sentiment/eval_data/aivivn.csv",
        "labels": ["positive", "negative"]  # 2 labels
    },
    "ntc_scv": {
        "path": "/workspace/adamergex-for-vietnamese-sentiment/eval_data/ntc_scv.csv",
        "labels": ["positive", "negative"]  # 2 labels
    },
    "viocd": {
        "path": "/workspace/adamergex-for-vietnamese-sentiment/eval_data/viocd.csv",
        "labels": ["positive", "negative"]  # 2 labels
    },
    "vlsp": {
        "path": "/workspace/adamergex-for-vietnamese-sentiment/eval_data/vlsp.csv",
        "labels": ["positive", "neutral", "negative"]  # 3 labels
    },
    "vsfc": {
        "path": "/workspace/adamergex-for-vietnamese-sentiment/eval_data/vsfc.csv",
        "labels": ["positive", "neutral", "negative"]  # 3 labels
    }
}

# Output directory for the merged adapter and results
MERGED_ADAPTER_SAVE_DIR = "./adapters/vi_sentiment_adamergex_xlmr_lora/"
RESULTS_FILE = os.path.join(MERGED_ADAPTER_SAVE_DIR, "evaluation_results.txt")

# Lambda hyperparameter for merging (tuned value, as per AdaMergeX)
LAMBDA_HYPER = 1.0

# Sentiment labels (consistent with train_adapter.py and process_data.py)
LABEL2ID_SENTIMENT = {"positive": 2, "neutral": 1, "negative": 0}
ID2LABEL_SENTIMENT = {v: k for k, v in LABEL2ID_SENTIMENT.items()}
NUM_LABELS_SENTIMENT = len(LABEL2ID_SENTIMENT)

# --- Helper Functions ---
def safe_create_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def compute_metrics_sentiment(eval_pred, num_labels):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    predictions, labels = eval_pred
    preds_label = np.argmax(predictions, axis=1)

    if num_labels == 2:
        # For binary classification, remove all label 1, which is neutral, replace by 0 if logit value at index 0 is higher than 2, or 2 if logit value at index 2 is higher
        for i in range(len(preds_label)):
            if preds_label[i] == 1:
                if predictions[i][0] > predictions[i][2]:
                    preds_label[i] = 0
                else:
                    preds_label[i] = 2
    # Compute metrics
    accuracy = accuracy_score(labels, preds_label)
    f1 = f1_score(labels, preds_label, average='weighted')
    
    # Generate classification report
    print ("mapped_labels: ", labels)
    print ("predictions: ", preds_label)
    report = classification_report(labels, preds_label, digits=4)
    
    return {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "classification_report": report
    }

def load_and_prepare_dataset(dataset_path, label_list, tokenizer):
    """Load and preprocess a CSV dataset for evaluation."""
    try:
        df = pd.read_csv(dataset_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"Dataset at {dataset_path} must have 'text' and 'label' columns")
        # Map text labels to indices
        # print ("BEFORE ----------------- ", df["label"].unique())
        df["labels"] = df["label"].map(LABEL2ID_SENTIMENT)
        # print ("AFTER ----------------- ", df["labels"].unique())
        if df["labels"].isnull().any():
            logging.warning(f"Found unmapped labels in {dataset_path}. Dropping invalid rows.")
            df = df.dropna(subset=["labels"])
        df["labels"] = df["labels"].astype(int)
        dataset = Dataset.from_pandas(df[["text", "labels"]])
        # Tokenize
        def tokenize_for_eval(examples):
            return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
        tokenized_dataset = dataset.map(tokenize_for_eval, batched=True, remove_columns=["text"])
        return tokenized_dataset, label_list
    except Exception as e:
        logging.error(f"Error loading dataset {dataset_path}: {e}")
        return None, None

# --- Main Evaluation Function ---
def merge_and_evaluate_adapters(
    adapter_base_dir,
    eng_sentiment_adapter_name,
    vi_ref_adapter_name,
    eng_ref_adapter_name,
    eval_datasets,
    merged_adapter_save_dir,
    lambda_hyper
):
    logging.info("Starting AdaMergeX adapter merging and evaluation for XLM-R.")
    safe_create_dir(merged_adapter_save_dir)

    # 1. Load Base Model and Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        base_model_for_sentiment = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME_OR_PATH,
            num_labels=NUM_LABELS_SENTIMENT,
            id2label=ID2LABEL_SENTIMENT,
            label2id=LABEL2ID_SENTIMENT,
            ignore_mismatched_sizes=True
        )
        logging.info(f"Loaded base model {MODEL_NAME_OR_PATH} for sentiment classification.")
    except Exception as e:
        logging.error(f"Error loading base model or tokenizer: {e}")
        return

    # 2. Load the Three LoRA Adapters Using PeftModel.from_pretrained
    path_eng_sentiment = os.path.join(adapter_base_dir, eng_sentiment_adapter_name)
    path_vi_ref = os.path.join(adapter_base_dir, vi_ref_adapter_name)
    path_eng_ref = os.path.join(adapter_base_dir, eng_ref_adapter_name)

    try:
        # Load PEFT models
        peft_model_eng_sentiment = PeftModel.from_pretrained(
            base_model_for_sentiment, path_eng_sentiment, is_trainable=False
        )
        # Load reference models with a base MLM model to avoid classifier head mismatch
        base_model_for_mlm = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME_OR_PATH,
            num_labels=NUM_LABELS_SENTIMENT,
            id2label=ID2LABEL_SENTIMENT,
            label2id=LABEL2ID_SENTIMENT,
            ignore_mismatched_sizes=True
        )
        peft_model_vi_ref = PeftModel.from_pretrained(
            base_model_for_mlm, path_vi_ref, is_trainable=False
        )
        peft_model_eng_ref = PeftModel.from_pretrained(
            base_model_for_mlm, path_eng_ref, is_trainable=False
        )

        
        # for name, param in peft_model_eng_sentiment.named_parameters():
        #     print(f"Adapter l2_t1 param: {name} - requires_grad: {param.requires_grad}")

        logging.info("Successfully loaded PEFT adapters.")
    except Exception as e:
        logging.error(f"Error loading PEFT adapters: {e}")
        return

    # 3. Perform Adapter Merging (AdaMergeX for LoRA)
    # A_l1t1 = A_l2t1 + lambda * (A_l1t2 - A_l2t2)
    merged_state_dict = {}
    sd_eng_sentiment = peft_model_eng_sentiment.state_dict()
    sd_vi_ref = peft_model_vi_ref.state_dict()
    sd_eng_ref = peft_model_eng_ref.state_dict()

    keys_eng_sentiment = set(sd_eng_sentiment.keys())
    keys_vi_ref = set(sd_vi_ref.keys())
    keys_eng_ref = set(sd_eng_ref.keys())
    all_keys = keys_eng_sentiment.union(keys_vi_ref).union(keys_eng_ref)

    for key in all_keys:
        val_eng_sent = sd_eng_sentiment.get(key)
        val_vi_ref = sd_vi_ref.get(key)
        val_eng_ref = sd_eng_ref.get(key)

        if val_eng_sent is None:
            logging.warning(f"Key {key} not found in English sentiment adapter. Skipping merge for this key.")
            continue

        is_lora_param = any(lora_part in key for lora_part in ['.lora_A.', '.lora_B.'])
        if is_lora_param:
            if val_vi_ref is not None and val_eng_ref is not None:
                merged_state_dict[key] = val_eng_sent + lambda_hyper * (val_vi_ref - val_eng_ref)
            else:
                logging.warning(f"LoRA key {key} missing in one of the reference adapters. Using Eng_Sentiment value.")
                merged_state_dict[key] = val_eng_sent
        elif "classifier" in key:
            merged_state_dict[key] = val_eng_sent
            logging.debug(f"Using original task head parameter for {key} from English Sentiment adapter.")
        else:
            logging.warning(f"Key {key} not identified as LoRA or classifier parameter. Using Eng_Sentiment value.")
            merged_state_dict[key] = val_eng_sent

    logging.info(f"Adapter parameters merged with lambda_hyper = {lambda_hyper}.")

    # 4. Load the Merged State Dict into a PeftModel
    try:
        peft_config_eng_sentiment = PeftConfig.from_pretrained(path_eng_sentiment)
        merged_peft_model = get_peft_model(base_model_for_sentiment, peft_config_eng_sentiment)
        merged_peft_model.load_state_dict(merged_state_dict, strict=False)
        logging.info("Merged state_dict loaded into PeftModel.")
    except Exception as e:
        logging.error(f"Error loading merged state_dict into PeftModel: {e}")
        return

    # 5. Save the Merged Adapter
    safe_create_dir(merged_adapter_save_dir)
    merged_peft_model.save_pretrained(merged_adapter_save_dir)
    tokenizer.save_pretrained(merged_adapter_save_dir)
    logging.info(f"Merged AdaMergeX adapter saved to {merged_adapter_save_dir}")

    # 6. Evaluate on Each Dataset
    eval_args = TrainingArguments(
        output_dir=os.path.join(merged_adapter_save_dir, "eval_output"),
        per_device_eval_batch_size=32,
        do_train=False,
        do_eval=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=merged_peft_model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    # Initialize results file
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("AdaMergeX XLM-R Evaluation Results for Vietnamese Sentiment Analysis\n")
        f.write(f"Lambda Hyperparameter: {lambda_hyper}\n\n")

    # Evaluate each dataset
    for dataset_name, dataset_info in eval_datasets.items():
        logging.info(f"Evaluating on {dataset_name}...")
        dataset_path = dataset_info["path"]
        label_list = dataset_info["labels"]
        num_labels = len(label_list)

        # Load and preprocess dataset
        tokenized_dataset, label_list = load_and_prepare_dataset(dataset_path, label_list, tokenizer)
        if tokenized_dataset is None:
            logging.error(f"Skipping evaluation for {dataset_name} due to loading error.")
            continue

        # Evaluate
        trainer.compute_metrics = lambda pred: compute_metrics_sentiment(pred, num_labels)
        eval_results = trainer.evaluate(tokenized_dataset)
        predictions = trainer.predict(tokenized_dataset)
        print (f"Predictions: {predictions.predictions}")
        print (f"Labels: {predictions.label_ids}")
        true_labels, pred_labels = predictions.label_ids, np.argmax(predictions.predictions, axis=1)

        if len(label_list) == 2:
            # For binary classification, remove all label 1, which is neutral, replace by 0 if logit value at index 0 is higher than 2, or 2 if logit value at index 2 is higher
            for i in range(len(pred_labels)):
                if pred_labels[i] == 1:
                    if predictions.predictions[i][0] > predictions.predictions[i][2]:
                        pred_labels[i] = 0
                    else:
                        pred_labels[i] = 2
        # Generate classification report
        report = classification_report(true_labels, pred_labels, digits=4)
        logging.info(f"Classification Report for {dataset_name}:\n{report}")

        # Save results
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n--- {dataset_name} ---\n")
            for key, value in eval_results.items():
                #print ("{key} {value:.4f}" , key , value)
                f.write(f"{key}: {value}\n")
            f.write("\nClassification Report:\n")
            f.write(report + "\n")

    logging.info(f"Evaluation results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and evaluate AdaMergeX adapters for XLM-R.")
    parser.add_argument("--adapter_dir", type=str, default=ADAPTER_BASE_DIR, help="Base directory where individual adapters are saved.")
    parser.add_argument("--eng_sent_name", type=str, default=ADAPTER_ENG_SENTIMENT_NAME, help="Name of the English sentiment adapter folder.")
    parser.add_argument("--vi_ref_name", type=str, default=ADAPTER_VI_REF_NAME, help="Name of the Vietnamese reference (MLM) adapter folder.")
    parser.add_argument("--eng_ref_name", type=str, default=ADAPTER_ENG_REF_NAME, help="Name of the English reference (MLM) adapter folder.")
    parser.add_argument("--merged_save_dir", type=str, default=MERGED_ADAPTER_SAVE_DIR, help="Directory to save the merged adapter and evaluation results.")
    parser.add_argument("--lambda_t", type=float, default=LAMBDA_HYPER, help="Lambda (t) hyperparameter for merging.")

    args = parser.parse_args()

    merge_and_evaluate_adapters(
        args.adapter_dir,
        args.eng_sent_name,
        args.vi_ref_name,
        args.eng_ref_name,
        EVAL_DATASETS,
        args.merged_save_dir,
        args.lambda_t
    )