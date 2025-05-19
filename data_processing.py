
import json
import random
from tqdm import tqdm
import os
import argparse
import pandas as pd  # For CSV handling
from transformers import AutoTokenizer  # For Llama tokenizer
from datasets import load_dataset 
# --- CLM Data Processing (Token-Based) ---
# Parameters for CLM data processing
CLM_CHUNK_SIZE = 512  # Number of tokens per chunk (prompt + completion)
CLM_COMPLETION_LENGTH = 1  # Number of tokens for completion
CLM_TRAIN_TEST_SPLIT_RATIO = 0.90  # Keep most for training
CLM_MAX_SAMPLES = 20000  # Limit samples, None for all
BATCH_SIZE = 1000  # Number of lines to process per batch for streaming
HUGGINGFACE_TOKEN = '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'  # Placeholder for Hugging Face token




def process_clm_data(raw_text_file_path: str, output_json_path: str, max_samples: int = None, hf_token: str = None):
    """
    Processes a raw text file into prompt/completion pairs for Causal Language Modeling using token-based splitting.
    Optimized for PEFT (e.g., LoRA) with SFTTrainer. Uses streaming to read the file line by line, tokenizes in batches,
    and pads incomplete chunks. Excludes special tokens in output for SFTTrainer compatibility.

    Output format: [{"prompt": "...", "completion": "..."}, ...]

    Args:
        raw_text_file_path (str): Path to the raw text file.
        output_json_path (str): Path to save the processed JSON output.
        max_samples (int, optional): Maximum number of samples to process.
        hf_token (str, optional): Hugging Face token for accessing gated models.
    """
    print(f"Processing CLM data from: {raw_text_file_path}")
    processed_instances = []

    # Load Llama-3.2-3B-Instruct tokenizer with authentication
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            token=hf_token
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Ensure you have requested access at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
        print("Log in with `huggingface-cli login` or pass a valid `hf_token`.")
        return

    # Set padding token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<|eot_id|>'  # Common fallback for Llama models
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        print("No pad token defined. Using '<|eot_id|>' as pad token.")

    # Initialize token buffer for streaming
    token_buffer = []
    line_count = 0
    batch_lines = []

    try:
        with open(raw_text_file_path, 'r', encoding='utf-8') as fin:
            for line in tqdm(fin, desc=f"Reading {os.path.basename(raw_text_file_path)}"):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                batch_lines.append(line)
                line_count += 1

                # Process batch when enough lines are collected
                if line_count % BATCH_SIZE == 0:
                    # Tokenize batch with special tokens
                    batch_tokens = tokenizer.encode(
                        '\n'.join(batch_lines),
                        add_special_tokens=True,
                        return_tensors='pt'
                    ).flatten().tolist()
                    token_buffer.extend(batch_tokens)
                    batch_lines = []  # Reset batch

                    # Process full chunks from token buffer
                    while len(token_buffer) >= CLM_CHUNK_SIZE:
                        chunk = token_buffer[:CLM_CHUNK_SIZE]
                        token_buffer = token_buffer[CLM_CHUNK_SIZE:]
                        process_chunk(chunk, tokenizer, processed_instances)

            # Process remaining lines after file is read
            if batch_lines:
                batch_tokens = tokenizer.encode(
                    '\n'.join(batch_lines),
                    add_special_tokens=True,
                    return_tensors='pt'
                ).flatten().tolist()
                token_buffer.extend(batch_tokens)

            # Process remaining tokens with padding
            while len(token_buffer):
                if len(token_buffer) >= CLM_COMPLETION_LENGTH:
                    chunk = token_buffer[:CLM_CHUNK_SIZE]
                    token_buffer = token_buffer[min(len(chunk), CLM_CHUNK_SIZE):]
                    if len(chunk) < CLM_CHUNK_SIZE:
                        # Pad chunk to CLM_CHUNK_SIZE
                        chunk.extend([tokenizer.pad_token_id] * (CLM_CHUNK_SIZE - len(chunk)))
                    process_chunk(chunk, tokenizer, processed_instances)
                else:
                    # Not enough tokens for completion; discard
                    break

    except Exception as e:
        print(f"Error reading raw text file {raw_text_file_path}: {e}")
        return

    if not processed_instances:
        print(f"No valid instances generated from {raw_text_file_path}.")
        return

    if max_samples and len(processed_instances) > max_samples:
        random.shuffle(processed_instances)
        processed_instances = processed_instances[:max_samples]

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding='utf-8') as fout:
        json.dump(processed_instances, fout, ensure_ascii=False, indent=4)
    print(f"CLM data processed and saved to: {output_json_path} ({len(processed_instances)} instances)")

def process_chunk(chunk, tokenizer, processed_instances):
    """Helper function to process a single chunk of tokens."""
    if len(chunk) != CLM_CHUNK_SIZE:
        return  # Skip invalid chunks
    prompt_tokens = chunk[:-CLM_COMPLETION_LENGTH]
    completion_tokens = chunk[-CLM_COMPLETION_LENGTH:]
    prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    if prompt_text and completion_text:
        processed_instances.append({
            "prompt": prompt_text,
            "completion": completion_text
        })

# --- Sentiment Analysis Data Processing ---

def process_sentiment_data(dataset_name: str = "cardiffnlp/tweet_eval", config_name: str = "sentiment", output_json_path: str = None, max_samples: int = None, hf_token: str = None):
    """
    Processes the tweet_eval sentiment dataset into a JSON format for SFTTrainer with PEFT.
    Loads the dataset using datasets library, maps labels to 'positive', 'negative', or 'neutral',
    and saves as a JSON file with instruction, input, and output fields.

    Args:
        dataset_name (str): Name of the dataset (default: 'cardiffnlp/tweet_eval').
        config_name (str): Dataset configuration (default: 'sentiment').
        output_json_path (str): Path to save the processed JSON output.
        max_samples (int, optional): Maximum number of samples to process.
        hf_token (str, optional): Hugging Face token (not used, included for compatibility).
    """
    print(f"Processing sentiment data from dataset: {dataset_name} ({config_name})")
    processed_instances = []

    # Label mapping for tweet_eval sentiment
    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, config_name, split="train")
        
        # Process each instance
        for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc=f"Formatting sentiment data from {dataset_name}"):
            if max_samples and idx >= max_samples:
                break
                
            instruction = "Analyze the sentiment of the following English text. Respond with only 'positive', 'negative', or 'neutral'."
            input_text = str(row['text'])
            output_label = label_map.get(row['label'], "unknown")
            
            if output_label == "unknown":
                print(f"Warning: Invalid label {row['label']} at index {idx}. Skipping instance.")
                continue

            processed_instances.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_label
            })

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        return

    if not processed_instances:
        print(f"No valid instances generated from {dataset_name}.")
        return

    if max_samples and len(processed_instances) > max_samples:
        random.shuffle(processed_instances)
        processed_instances = processed_instances[:max_samples]

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding='utf-8') as fout:
        json.dump(processed_instances, fout, ensure_ascii=False, indent=4)
    print(f"Sentiment data processed and saved to: {output_json_path} ({len(processed_instances)} instances)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for AdaMergeX training with PEFT.")
    parser.add_argument("--task_type", required=True, choices=["clm", "sentiment"], help="Type of data to process.")
    parser.add_argument("--input_file", required=False, help="Path to the raw input file (txt for clm, ignored for sentiment).")
    parser.add_argument("--output_file", required=True, help="Path to save the processed JSON output file.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process and save.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for accessing gated models.")

    args = parser.parse_args()

    if args.task_type == "clm":
        if not args.input_file:
            print("Error: --input_file is required for task_type 'clm'")
            exit(1)
        process_clm_data(args.input_file, args.output_file, args.max_samples, args.hf_token)
    elif args.task_type == "sentiment":
        process_sentiment_data(
            dataset_name="cardiffnlp/tweet_eval",
            config_name="sentiment",
            output_json_path=args.output_file,
            max_samples=args.max_samples,
            hf_token=args.hf_token
        )
# # data_processing.py
# import json
# import random
# from tqdm import tqdm
# import os
# import argparse
# import pandas as pd # For CSV handling

# # --- CLM Data Processing (inspired by construct_dataset_lm.py) ---
# # Parameters for CLM data processing
# CLM_CHUNK_SIZE = 512  # As implied by filenames like Spanish_Wiki_10k_LM_1_511_test.json
# CLM_COMPLETION_LENGTH = 1 # To make prompt 511 and completion 1
# CLM_TRAIN_TEST_SPLIT_RATIO = 0.90 # Keep most for training
# CLM_MAX_SAMPLES = 20000 # Set to an int (e.g., 20000) to limit, None for all

# def process_clm_data(raw_text_file_path: str, output_json_path: str, max_samples: int = None):
#     """
#     Processes a raw text file into prompt/completion pairs for Causal Language Modeling.
#     Saves the output as a JSON file compatible with SFTTrainer (list of dicts with 'text' field).
#     The 'text' field will contain "```{prompt}```{completion}"
#     """
#     print(f"Processing CLM data from: {raw_text_file_path}")
#     processed_instances = []
#     full_text_content = ""
#     try:
#         with open(raw_text_file_path, 'r', encoding='utf-8') as fin:
#             full_text_content = fin.read()
#     except Exception as e:
#         print(f"Error reading raw text file {raw_text_file_path}: {e}")
#         return

#     if not full_text_content:
#         print(f"Raw text file {raw_text_file_path} is empty.")
#         return

#     # Similar logic to the original construct_dataset_lm.py's character-based chunking
#     current_text_segment = ""
#     for char_idx, char in tqdm(enumerate(full_text_content), desc=f"Generating CLM instances from {os.path.basename(raw_text_file_path)}"):
#         current_text_segment += char
#         if len(current_text_segment) == CLM_CHUNK_SIZE:
#             prompt_part = current_text_segment[:-CLM_COMPLETION_LENGTH]
#             completion_part = current_text_segment[-CLM_COMPLETION_LENGTH:]
#             if prompt_part and completion_part:
#                 # Format for SFTTrainer (single 'text' field)
#                 # This matches the formatting_prompts_func from the original train_lora.py
#                 formatted_text = f"```{prompt_part}```{completion_part}"
#                 processed_instances.append({"text": formatted_text})
#             current_text_segment = "" # Reset

#     if max_samples and len(processed_instances) > max_samples:
#         random.shuffle(processed_instances) # Shuffle before sampling
#         processed_instances = processed_instances[:max_samples]

#     os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
#     with open(output_json_path, "w", encoding='utf-8') as fout:
#         # SFTTrainer usually expects a list of dicts, or a Dataset object.
#         # Saving as JSON list of dicts.
#         json.dump(processed_instances, fout, ensure_ascii=False, indent=4)
#     print(f"CLM data processed and saved to: {output_json_path} ({len(processed_instances)} instances)")

# # --- Sentiment Analysis Data Processing ---
# def process_sentiment_data(input_file_path: str, output_json_path: str, max_samples: int = None):
#     """
#     Processes sentiment data (e.g., from CSV) into a JSON format for SFTTrainer.
#     Expects input_file to have 'text' and 'sentiment_label' columns.
#     Saves as a JSON file (list of dicts with 'text' field for Llama 3 Instruct).
#     """
#     print(f"Processing sentiment data from: {input_file_path}")
#     processed_instances = []
#     try:
#         if input_file_path.endswith(".csv"):
#             df = pd.read_csv(input_file_path)
#         elif input_file_path.endswith(".json") or input_file_path.endswith(".jsonl"):
#             # Assuming jsonl where each line is a dict or a json list of dicts
#             temp_data = []
#             with open(input_file_path, 'r', encoding='utf-8') as f:
#                 if input_file_path.endswith(".jsonl"):
#                     for line in f:
#                         temp_data.append(json.loads(line))
#                 else: # .json
#                     temp_data = json.load(f)
#             df = pd.DataFrame(temp_data)
#         else:
#             print(f"Unsupported file format for sentiment data: {input_file_path}")
#             return

#         # Ensure required columns exist
#         if 'text_input' not in df.columns or 'sentiment_label' not in df.columns:
#              # Try common alternatives if direct match fails
#             if 'text' in df.columns and 'label' in df.columns:
#                 df.rename(columns={'text': 'text_input', 'label': 'sentiment_label'}, inplace=True)
#             elif 'review' in df.columns and 'sentiment' in df.columns:
#                  df.rename(columns={'review': 'text_input', 'sentiment': 'sentiment_label'}, inplace=True)
#             else:
#                 print(f"Error: Input file {input_file_path} must contain 'text_input' and 'sentiment_label' columns (or common alternatives like 'text'/'label').")
#                 print(f"Found columns: {df.columns.tolist()}")
#                 return


#         for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Formatting sentiment data from {os.path.basename(input_file_path)}"):
#             instruction = "Analyze the sentiment of the following text. Respond with only 'positive', 'negative', or 'neutral'."
#             input_text = str(row['text_input'])
#             output_label = str(row['sentiment_label']).lower() # Ensure consistency

#             # Llama 3 Instruct format
#             formatted_text = (
#                 f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
#                 f"<|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|>"
#                 f"<|start_header_id|>assistant<|end_header_id|>\n\n{output_label}<|eot_id|>"
#             )
#             processed_instances.append({"text": formatted_text})

#     except Exception as e:
#         print(f"Error processing sentiment file {input_file_path}: {e}")
#         return

#     if max_samples and len(processed_instances) > max_samples:
#         random.shuffle(processed_instances) # Shuffle before sampling
#         processed_instances = processed_instances[:max_samples]

#     os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
#     with open(output_json_path, "w", encoding='utf-8') as fout:
#         json.dump(processed_instances, fout, ensure_ascii=False, indent=4)
#     print(f"Sentiment data processed and saved to: {output_json_path} ({len(processed_instances)} instances)")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process data for AdaMergeX training.")
#     parser.add_argument("--task_type", required=True, choices=["clm", "sentiment"], help="Type of data to process.")
#     parser.add_argument("--input_file", required=True, help="Path to the raw input file (txt for clm, csv/json/jsonl for sentiment).")
#     parser.add_argument("--output_file", required=True, help="Path to save the processed JSON output file.")
#     parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process and save.")
#     # For CLM data, you could add arguments for chunk_size, completion_length if needed

#     args = parser.parse_args()

#     if args.task_type == "clm":
#         process_clm_data(args.input_file, args.output_file, args.max_samples)
#     elif args.task_type == "sentiment":
#         process_sentiment_data(args.input_file, args.output_file, args.max_samples)

#     # --- Example Usage (run these from your terminal) ---
#     # Create dummy raw data files first for testing

#     # print("Creating dummy raw_vietnamese_clm.txt for CLM processing example...")
#     # with open("raw_vietnamese_clm.txt", "w", encoding="utf-8") as f:
#     #     for i in range(200): # Create enough text for a few chunks
#     #         f.write(f"Đây là một câu tiếng Việt mẫu số {i} để thử nghiệm việc tạo dữ liệu cho mô hình ngôn ngữ nhân quả. ")
#     #         f.write("Mục tiêu là cắt văn bản này thành các đoạn nhỏ hơn với prompt và completion. " * 5 + "\n")
#     #
#     # print("Creating dummy raw_english_sentiment.csv for Sentiment processing example...")
#     # dummy_sentiment_data = {
#     #     'text_input': ["I love this product, it's amazing!", "This is the worst experience ever.", "It's okay, not great but not bad."],
#     #     'sentiment_label': ["positive", "negative", "neutral"]
#     # }
#     # pd.DataFrame(dummy_sentiment_data).to_csv("raw_english_sentiment.csv", index=False)

#     # print("\nTo run processing:")
#     # print("python data_processing.py --task_type clm --input_file raw_vietnamese_clm.txt --output_file ./processed_data/vietnamese_clm_train.json --max_samples 100")
#     # print("python data_processing.py --task_type sentiment --input_file raw_english_sentiment.csv --output_file ./processed_data/english_sentiment_train.json")

#     # print("\nNote: Adjust the paths and filenames as needed.")
#     # print("You can also use the --max_samples argument to limit the number of samples processed.")
#     # print("Ensure you have the required libraries installed: tqdm, pandas.")