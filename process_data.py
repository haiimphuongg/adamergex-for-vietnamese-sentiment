import hashlib
import json
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import logging
import random
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# File paths
VI_MLM_RAW_PATH = "./data/raw/vietnamese_mlm_raw.txt"  # Vietnamese MLM data (250k lines)
EN_MLM_RAW_PATH = "./data/raw/en_mlm_raw.txt"          # English MLM data (2M lines)
SENTIMENT_RAW_PATH = "/workspace/adamergex-for-vietnamese-sentiment/processed_data/english_sentiment_corpus.json"  # Sentiment data (JSONL, both languages)
VI_MLM_PROCESSED_DIR = "./data/processed/vi_mlm/"
EN_MLM_PROCESSED_DIR = "./data/processed/en_mlm/"
VI_SENTIMENT_PROCESSED_DIR = "./data/processed/vi_sentiment/"
EN_SENTIMENT_PROCESSED_DIR = "./data/processed/en_sentiment/"

# Label mapping for sentiment
SENTIMENT_LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

# MLM settings
MAX_TOKEN_LENGTH = 128  # Max token length for MLM samples
TARGET_SAMPLES = 100_000  # Target number of samples for MLM datasets
VAL_SIZE = 0.05  # 5% validation split for MLM
TEST_SIZE = 0.1  # 10% test split for sentiment (5% validation, 5% test)

# Initialize XLM-RoBERTa tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
except Exception as e:
    logging.error(f"Error loading XLM-RoBERTa tokenizer: {e}")
    raise

# --- Helper Functions ---
def safe_create_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def filter_by_token_length(texts, max_tokens=MAX_TOKEN_LENGTH):
    """Filter texts to those with <= max_tokens using XLM-RoBERTa tokenizer."""
    filtered_texts = []
    for text in tqdm(texts, desc="Filtering by token length"):
        tokens = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if len(tokens) <= max_tokens:
            filtered_texts.append(text)
    return filtered_texts

def upsample_to_target(texts, target_size=TARGET_SAMPLES):
    """Upsample texts with replacement to reach target_size."""
    if len(texts) >= target_size:
        return texts[:target_size]
    logging.info(f"Upsampling from {len(texts)} to {target_size} samples")
    return texts + random.choices(texts, k=target_size - len(texts))

def process_mlm_data(raw_path, processed_dir, lang_prefix):
    """Loads unlabeled text data, filters by token length, upsamples, splits, and saves for MLM."""
    safe_create_dir(processed_dir)
    logging.info(f"Processing MLM data from: {raw_path}")

    try:
        # Read text file
        with open(raw_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            logging.warning(f"No text found in {raw_path}")
            return

        # Deduplicate
        seen_hashes = set()
        unique_lines = []
        for line in tqdm(lines, desc="Deduplicating"):
            line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
            if line_hash not in seen_hashes:
                seen_hashes.add(line_hash)
                unique_lines.append(line)

        # Filter by token length
        filtered_lines = filter_by_token_length(unique_lines)
        logging.info(f"After filtering, {len(filtered_lines)} samples remain")

        # Upsample to target size
        final_lines = upsample_to_target(filtered_lines)
        logging.info(f"Final sample count: {len(final_lines)}")

        # Split data (95% train, 5% validation)
        train_lines, val_lines = train_test_split(final_lines, test_size=VAL_SIZE, random_state=42)

        # Create Hugging Face Dataset
        train_dataset = Dataset.from_dict({"text": train_lines})
        val_dataset = Dataset.from_dict({"text": val_lines})
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        dataset_dict.save_to_disk(processed_dir)
        logging.info(f"Processed MLM data saved to: {processed_dir}")

    except Exception as e:
        logging.error(f"Error processing MLM data {raw_path}: {e}")

def process_sentiment_data(raw_path, processed_dir, label_map, lang_prefix, val_size=0.05):
    """Loads sentiment JSON data, maps labels, splits into train/val, and saves as Hugging Face Dataset."""
    safe_create_dir(processed_dir)
    logging.info(f"Processing sentiment data from: {raw_path}")

    try:
        # Read JSON file (not JSONL, since the file is a JSON array)
        df = pd.read_json(raw_path)
        if "input" not in df.columns or "output" not in df.columns:
            raise ValueError("Sentiment data must contain 'input' and 'output' columns")

        # Filter by language if specified
        if lang_prefix != "both":
            if "language" in df.columns:
                df = df[df["language"] == lang_prefix]
                if df.empty:
                    logging.warning(f"No {lang_prefix} data found in {raw_path}")
                    return
            else:
                logging.warning(f"No 'language' column in {raw_path}; processing all data")

        # Map labels to integers
        df["labels"] = df["output"].map(label_map)
        if df["labels"].isnull().any():
            logging.warning(f"Found unmapped labels in {raw_path}. Check SENTIMENT_LABEL_MAP: {set(df['output'])}")
            df.dropna(subset=["labels"], inplace=True)
        df["labels"] = df["labels"].astype(int)

        # Rename 'input' to 'text'
        df = df[["input", "labels"]].rename(columns={"input": "text"})

        # Split data (90% train, 10% validation)
        if df.shape[0] < 100:
            train_df = df
            val_df = df.sample(frac=0.1, random_state=42) if df.shape[0] > 10 else df
        else:
            train_df, val_df = train_test_split(df, test_size=val_size, random_state=42, stratify=df["labels"])

        # Create Hugging Face Dataset
        dataset_dict_content = {}
        if not train_df.empty:
            dataset_dict_content["train"] = Dataset.from_pandas(train_df[["text", "labels"]])
        if not val_df.empty:
            dataset_dict_content["validation"] = Dataset.from_pandas(val_df[["text", "labels"]])

        if not dataset_dict_content:
            logging.warning(f"No data produced for {processed_dir}. Check input file and splits.")
            return

        dataset_dict = DatasetDict(dataset_dict_content)
        dataset_dict.save_to_disk(processed_dir)
        logging.info(f"Processed sentiment data saved to: {processed_dir}")

    except Exception as e:
        logging.error(f"Error processing sentiment data {raw_path}: {e}")
        raise

# --- Main Execution ---
if __name__ == "__main__":
    # #1. Process Vietnamese MLM Data
    # if os.path.exists(VI_MLM_RAW_PATH):
    #     process_mlm_data(VI_MLM_RAW_PATH, VI_MLM_PROCESSED_DIR, "vi")
    # else:
    #     logging.warning(f"Vietnamese MLM raw data not found at {VI_MLM_RAW_PATH}, skipping.")

    # # 2. Process English MLM Data
    # if os.path.exists(EN_MLM_RAW_PATH):
    #     process_mlm_data(EN_MLM_RAW_PATH, EN_MLM_PROCESSED_DIR, "en")
    # else:
    #     logging.warning(f"English MLM raw data not found at {EN_MLM_RAW_PATH}, skipping.")

    # 3. Process Sentiment Data (English)
    #process_sentiment_data(SENTIMENT_RAW_PATH, EN_SENTIMENT_PROCESSED_DIR, SENTIMENT_LABEL_MAP, "en")
    if os.path.exists(SENTIMENT_RAW_PATH):
        process_sentiment_data(SENTIMENT_RAW_PATH, EN_SENTIMENT_PROCESSED_DIR, SENTIMENT_LABEL_MAP, "en")
    else:
        logging.warning(f"English sentiment raw data not found at {SENTIMENT_RAW_PATH}, skipping.")

    logging.info("Data processing finished.")
