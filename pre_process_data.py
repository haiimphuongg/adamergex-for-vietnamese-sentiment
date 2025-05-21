# import hashlib
# import os
# from datasets import load_dataset
# from tqdm import tqdm
# import nltk

# # Download NLTK punkt and punkt_tab for sentence tokenization
# nltk.download('punkt')
# nltk.download('punkt_tab')
# from nltk.tokenize import sent_tokenize

# def hash_sentence(sentence):
#     """Create a memory-efficient hash for deduplication."""
#     return hashlib.md5(sentence.encode('utf-8')).hexdigest()

# def estimate_file_size(output_file):
#     """Estimate the size of the output file in GB."""
#     if os.path.exists(output_file):
#         return os.path.getsize(output_file) / (1024 ** 3)  # Convert bytes to GB
#     return 0

# def extract_cc_news_for_mlm(output_txt, target_sentences=200_000):
#     """
#     Stream CC-News dataset, extract 200,000 unique sentences for MLM training,
#     and save to TXT. Uses streaming to avoid memory overload.
    
#     Args:
#         output_txt (str): Path to the output text file (e.g., 'en_mlm_raw.txt').
#         target_sentences (int): Number of unique sentences to extract (default: 200,000).
#     """
#     # Load CC-News dataset in streaming mode (only train split available)
#     dataset = load_dataset("vblagoje/cc_news", split="train", streaming=True)
    
#     # Set to store sentence hashes (for deduplication)
#     seen_hashes = set()
#     collected_sentences = 0
    
#     # Ensure output directory exists
#     output_dir = os.path.dirname(output_txt)
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"Created directory: {output_dir}")
    
#     # Open TXT file for writing
#     with open(output_txt, 'w', encoding='utf-8') as f_out:
#         # Initialize progress bar
#         with tqdm(total=target_sentences, desc="Extracting sentences for MLM") as pbar:
#             for example in dataset:
#                 # Extract text field and tokenize into sentences
#                 text = example['text'].strip()
#                 if not text:
#                     continue
#                 sentences = sent_tokenize(text)
                
#                 for sentence in sentences:
#                     sentence = sentence.strip()
#                     # Skip empty, short (<10 chars), or overly long (>500 chars) sentences
#                     if not sentence or len(sentence) < 10 or len(sentence) > 500:
#                         continue
                    
#                     # Compute hash for deduplication
#                     sentence_hash = hash_sentence(sentence)
                    
#                     # Check if sentence is unique
#                     if sentence_hash not in seen_hashes:
#                         seen_hashes.add(sentence_hash)
#                         f_out.write(sentence + '\n')
#                         collected_sentences += 1
                        
#                         # Update progress
#                         pbar.update(1)
                        
#                         # Stop if target sentences reached
#                         if collected_sentences >= target_sentences:
#                             break
                
#                 # Break if target reached
#                 if collected_sentences >= target_sentences:
#                     break
    
#     final_size = estimate_file_size(output_txt)
#     print(f"Extracted {collected_sentences} unique sentences to {output_txt} ({final_size:.2f} GB)")

# if __name__ == "__main__":
#     # Output file compatible with process_data.py
#     output_txt = "./data/raw/en_mlm_raw.txt"  # Path for MLM raw data
#     target_sentences = 2_000_000  # Target number of sentences
    
#     # Extract sentences for MLM training
#     extract_cc_news_for_mlm(output_txt, target_sentences=target_sentences)


import json
import os

input_file = "/workspace/adamergex-for-vietnamese-sentiment/processed_data/vietnamese_clm_corpus.json"
output_file = "/workspace/adamergex-for-vietnamese-sentiment/data/raw/vietnamese_mlm_raw.txt"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Read and process the JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    for item in data:
        f.write(f"{item['prompt']}{item['completion']}\n")

print("✅ File created:", output_file)
