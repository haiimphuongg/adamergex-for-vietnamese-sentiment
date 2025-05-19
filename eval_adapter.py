import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import argparse
import re # For simple answer extraction in demo

# Default model and tokenizer - can be overridden by args
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
TOKENIZER_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_LAMBDA_HYPER = 1.0
MAX_SEQ_LENGTH_INFERENCE = 1024 # Max sequence length for inference prompt

def load_base_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path, for_merging=False):
    print(f"Loading tokenizer: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Llama 3 uses eos_token as pad
    # tokenizer.padding_side = "right" # Default for Llama typically

    print(f"Loading base model: {model_name_or_path}")
    device_map_config = "cpu" if for_merging else "auto" # Load to CPU for merging ops if VRAM is a concern

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map_config,
        trust_remote_code=True
    )
    return model, tokenizer

def main(args):
    # --- 1. Load Base Model and Tokenizer ---
    # We need three instances of the base model to load adapters separately,
    # as in the provided test_combine_llama2.py, to avoid issues with
    # PeftModel modifying the base model instance when adapters are loaded.
    # Alternatively, load one base, then attach adapters one by one to separate PeftModel instances if memory is tight.
    # For direct adaptation of test_combine_llama2.py's logic:
    print("Loading base models (this might take a while and consume VRAM/RAM)...")
    base_model_instance1, tokenizer = load_base_model_and_tokenizer(args.base_model_name, args.tokenizer_name, for_merging=True)
    base_model_instance2, _ = load_base_model_and_tokenizer(args.base_model_name, args.tokenizer_name, for_merging=True)
    base_model_instance3, _ = load_base_model_and_tokenizer(args.base_model_name, args.tokenizer_name, for_merging=True)

    # --- 2. Load Adapters ---
    print(f"Loading adapter for Target Task in Source Lang (l2_t1) from: {args.adapter_l2_t1_path}")
    peft_model_l2_t1 = PeftModel.from_pretrained(base_model_instance1, args.adapter_l2_t1_path, adapter_name=args.adapter_name_l2_t1)
    
    print(f"Loading adapter for Reference Task in Target Lang (l1_t2) from: {args.adapter_l1_t2_path}")
    peft_model_l1_t2 = PeftModel.from_pretrained(base_model_instance2, args.adapter_l1_t2_path, adapter_name=args.adapter_name_l1_t2)

    print(f"Loading adapter for Reference Task in Source Lang (l2_t2) from: {args.adapter_l2_t2_path}")
    peft_model_l2_t2 = PeftModel.from_pretrained(base_model_instance3, args.adapter_l2_t2_path, adapter_name=args.adapter_name_l2_t2)

    # for name, param in peft_model_l2_t1.named_parameters():
    #     if args.adapter_name_l2_t1 in name:
    #         print(f"Adapter l2_t1 param: {name} - requires_grad: {param.requires_grad}")
    # --- 3. Extract Parameters (as in test_combine_llama2.py) ---
    # Ensure parameters are on CPU for manipulation if device_map="cpu" was used
    params_l2_t1 = {name: param.to("cpu").clone() for name, param in peft_model_l2_t1.named_parameters() if args.adapter_name_l2_t1 in name}
    params_l1_t2 = {name: param.to("cpu").clone() for name, param in peft_model_l1_t2.named_parameters() if args.adapter_name_l1_t2 in name}
    params_l2_t2 = {name: param.to("cpu").clone() for name, param in peft_model_l2_t2.named_parameters() if args.adapter_name_l2_t2 in name}

    if not params_l2_t1 or not params_l1_t2 or not params_l2_t2:
        print("Error: Could not extract LoRA parameters from one or more adapters. Check adapter names and paths.")
        print(f"Keys found for l2_t1 ({args.adapter_name_l2_t1}): {list(params_l2_t1.keys())}")
        print(f"Keys found for l1_t2 ({args.adapter_name_l1_t2}): {list(params_l1_t2.keys())}")
        print(f"Keys found for l2_t2 ({args.adapter_name_l2_t2}): {list(params_l2_t2.keys())}")
        return

    # --- 4. Calculate Language Ability ($A_{l1,t2} - A_{l2,t2}$) ---
    # This modifies params_l1_t2 in place to become the language_ability_delta
    print("Calculating language ability vector (modifies params for adapter l1_t2)...")
    lang_ability_keys_map = {} # To store the modified params_l1_t2 which represent language ability

    with torch.no_grad():
        for name_l1t2, param_l1t2_val in tqdm(params_l1_t2.items(), desc="Calculating Language Delta"):
            # Derive corresponding name in l2_t2 adapter
            # e.g. base_model...<adapter_name_l1_t2>.lora... -> base_model...<adapter_name_l2_t2>.lora...
            name_l2t2_equiv = name_l1t2.replace(args.adapter_name_l1_t2, args.adapter_name_l2_t2)
            param_l2t2_val = params_l2_t2.get(name_l2t2_equiv, None)

            if param_l2t2_val is not None:
                # Modify the param_l1t2_val in params_l1_t2 directly
                params_l1_t2[name_l1t2].data = param_l1t2_val.data - param_l2t2_val.data
                lang_ability_keys_map[name_l1t2] = params_l1_t2[name_l1t2].data # Store the delta
            else:
                print(f"Warning: Parameter {name_l2t2_equiv} (derived from {name_l1t2}) not found in adapter '{args.adapter_name_l2_t2}'. Skipping subtraction for this param.")
                lang_ability_keys_map[name_l1t2] = params_l1_t2[name_l1t2].data # Store original if counterpart not found

    # At this point, `params_l1_t2` (or rather, `lang_ability_keys_map`) effectively holds the "language ability" delta.

    # --- 5. Calculate Final Merged Adapter ($A_{l2,t1} + \lambda \cdot \text{language\_ability\_delta}$) ---
    # This modifies params_l2_t1 in place to become the final merged adapter weights.
    print("Calculating final merged adapter weights (modifies params for adapter l2_t1)...")
    with torch.no_grad():
        for name_l2t1, param_l2t1_val in tqdm(params_l2_t1.items(), desc="Merging Adapters"):
            # Derive corresponding name in the language_ability_delta map (which used l1_t2 names)
            name_lang_ability_equiv = name_l2t1.replace(args.adapter_name_l2_t1, args.adapter_name_l1_t2)
            lang_ability_tensor = lang_ability_keys_map.get(name_lang_ability_equiv, None)

            if lang_ability_tensor is not None:
                params_l2_t1[name_l2t1].data = param_l2t1_val.data + args.lambda_hyper * lang_ability_tensor.data
            else:
                print(f"Warning: Language ability tensor for {name_lang_ability_equiv} (derived from {name_l2t1}) not found. Skipping addition for this param.")

    # --- 6. Update PeftModel with Merged Weights and Save ---
    # The `peft_model_l2_t1`'s adapter weights (named `args.adapter_name_l2_t1`) are now the merged ones.
    # We need to update the actual model's parameters.
    # The parameters in params_l2_t1 are clones; we need to write them back.
    print(f"Updating PeftModel '{args.adapter_name_l2_t1}' with merged weights...")
    state_dict_to_apply = {}
    for name, modified_param_tensor in params_l2_t1.items():
        state_dict_to_apply[name] = modified_param_tensor

    peft_model_l2_t1.load_state_dict(state_dict_to_apply, strict=False) # strict=False to ignore non-adapter params

    # The `peft_model_l2_t1` now contains the merged adapter.
    merged_adapter_final_path = args.merged_adapter_save_path
    print(f"Saving merged adapter to: {merged_adapter_final_path}")
    os.makedirs(merged_adapter_final_path, exist_ok=True)
    # Save only the adapter part that was modified (which is named args.adapter_name_l2_t1 on this peft_model object)
    peft_model_l2_t1.save_pretrained(merged_adapter_final_path, selected_adapters=[args.adapter_name_l2_t1])
    # It's good practice to save the tokenizer with the adapter
    tokenizer.save_pretrained(merged_adapter_final_path)
    print(f"Merged adapter (originally '{args.adapter_name_l2_t1}', now merged) saved.")

    # # --- 7. Inference with the Merged Adapter ---
    # print("\n--- Performing Inference with Merged Adapter ---")
    # # Load a fresh base model and the *saved* merged adapter for a clean inference setup
    # inference_base_model, inference_tokenizer = load_base_model_and_tokenizer(args.base_model_name, args.tokenizer_name, for_merging=False) # Load to GPU

    # print(f"Loading merged adapter from: {merged_adapter_final_path} for inference.")
    # # When PeftModel.from_pretrained loads a saved adapter, if only one adapter was saved (selected_adapters),
    # # it often loads it as 'default', or with the name from its adapter_config.json.
    # # The save_pretrained with selected_adapters might save it as 'args.adapter_name_l2_t1'.
    # # Let's try loading with the original name it was saved under from the selection.
    # try:
    #     # Check the adapter_config.json in merged_adapter_final_path to see what name it was saved as.
    #     # If selected_adapters=["name"], it saves config for "name".
    #     final_inference_model = PeftModel.from_pretrained(inference_base_model, merged_adapter_final_path + "/en_sentiment_adapter") # Default load
    # except Exception as e:
    #     print(f"Could not load merged adapter with default name. Error: {e}")
    #     print(f"Try specifying adapter_name='{args.adapter_name_l2_t1}' if that's how it was saved.")
    #     try:
    #         final_inference_model = PeftModel.from_pretrained(inference_base_model, merged_adapter_final_path, adapter_name=args.adapter_name_l2_t1)
    #     except Exception as e2:
    #         print(f"Failed to load adapter even with name {args.adapter_name_l2_t1}. Error: {e2}")
    #         return

    # final_inference_model.to("cuda" if torch.cuda.is_available() else "cpu") # Ensure on GPU
    # final_inference_model.eval()

    # if args.test_text_vi:
    #     # Ensure the correct adapter is active on the model for inference
    #     # If only one adapter was saved and loaded, it's usually active by default or named 'default'.
    #     # If it was loaded with a specific name, set it active.
    #     try:
    #         active_adapters = final_inference_model.active_adapters
    #         print(f"Active adapters on loaded model: {active_adapters}")
    #         if not active_adapters: # if empty, try to set the one we think was loaded
    #              final_inference_model.set_adapter(args.adapter_name_l2_t1 if args.adapter_name_l2_t1 in final_inference_model.peft_config else "default")
    #         elif len(active_adapters) > 1: # if multiple, select the one
    #              final_inference_model.set_adapter(args.adapter_name_l2_t1 if args.adapter_name_l2_t1 in final_inference_model.peft_config else "default")

    #     except Exception as e:
    #         print(f"Could not set active adapter, attempting inference with current state. Error: {e}")


    #     prompt = (
    #         f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    #         f"Analyze the sentiment of the following Vietnamese text. Respond with only 'positive', 'negative', or 'neutral'.<|eot_id|>"
    #         f"<|start_header_id|>user<|end_header_id|>\n\n{args.test_text_vi}<|eot_id|>"
    #         f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    #     )
    #     inputs = inference_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH_INFERENCE - 32).to(final_inference_model.device)
    #     print(f"\nInput to model (decoded for check):\n{inference_tokenizer.decode(inputs['input_ids'][0])}")

    #     with torch.no_grad():
    #         outputs = final_inference_model.generate(
    #             **inputs,
    #             max_new_tokens=10, # positive, negative, neutral
    #             eos_token_id=[inference_tokenizer.eos_token_id, inference_tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    #             pad_token_id=inference_tokenizer.eos_token_id,
    #             do_sample=False,
    #         )
        
    #     response_ids = outputs[0]
    #     sentiment = inference_tokenizer.decode(response_ids, skip_special_tokens=True).strip().lower()
        
    #     print(f"\nVietnamese Text: '{args.test_text_vi}'")
    #     print(f"Predicted Sentiment (AdaMergeX): '{sentiment}'")
    # --- 7. Inference with the Merged Adapter ---
    print("\n--- Performing Inference with Merged Adapter ---")
    # Load a fresh base model and the *saved* merged adapter for a clean inference setup
    inference_base_model, inference_tokenizer = load_base_model_and_tokenizer(args.base_model_name, args.tokenizer_name, for_merging=False) # Load to GPU

    print(f"Loading merged adapter from: {merged_adapter_final_path} for inference.")
    try:
        # Load the merged adapter, assuming it was saved with args.adapter_name_l2_t1
        final_inference_model = PeftModel.from_pretrained(
            inference_base_model,
            merged_adapter_final_path + "/en_sentiment_adapter",  # Adjust this if the adapter was saved with a different name
            is_trainable=False,  # Inference mode
        )
    except Exception as e:
        print(f"Failed to load merged adapter. Error: {e}")
        return

    # Ensure model is on the correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_inference_model.to(device)
    final_inference_model.eval()

    # Verify and set the active adapter
    try:
        active_adapters = final_inference_model.active_adapters()
        print(f"Active adapters on loaded model: {active_adapters}")
        expected_adapter = args.adapter_name_l2_t1
        if expected_adapter in final_inference_model.peft_config:
            final_inference_model.set_adapter(expected_adapter)
        elif "default" in final_inference_model.peft_config and len(final_inference_model.peft_config) == 1:
            final_inference_model.set_adapter("default")
            print(f"Using 'default' adapter as only one adapter is present.")
        else:
            print(f"Warning: Adapter '{expected_adapter}' not found. Available adapters: {list(final_inference_model.peft_config.keys())}")
            print("Proceeding with current active adapter(s), which may lead to incorrect results.")
    except Exception as e:
        print(f"Error setting active adapter: {e}")
        print("Proceeding with current adapter state.")

    if args.test_text_vi:
        # Construct prompt for sentiment analysis
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id>\n\n"
            f"Analyze the sentiment of the following Vietnamese text. Respond with only 'positive', 'negative', or 'neutral'.<|eot_id>"
            f"<|start_header_id|>user<|end_header_id>\n\n{args.test_text_vi}<|eot_id>"
            f"<|start_header_id|>assistant<|end_header_id>\n\n"
        )
        inputs = inference_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH_INFERENCE - 32,
            padding=True
        ).to(device)
        print(f"\nInput to model (decoded for check):\n{inference_tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}")

        with torch.no_grad():
            outputs = final_inference_model.generate(
                **inputs,
                max_new_tokens=20,  # Increased slightly to allow for potential extra tokens
                eos_token_id=[inference_tokenizer.eos_token_id, inference_tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                pad_token_id=inference_tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,  # Greedy decoding
                return_dict_in_generate=True,
                output_scores=False
            )

        # Extract generated token IDs (exclude the input prompt)
        generated_ids = outputs.sequences[0, inputs['input_ids'].shape[1]:]
        sentiment = inference_tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()

        # Validate sentiment output
        valid_sentiments = ['positive', 'negative', 'neutral']
        if sentiment not in valid_sentiments:
            print(f"Warning: Output '{sentiment}' is not a valid sentiment. Raw output may contain extra tokens.")
            # Attempt to extract the first valid sentiment word
            for valid_sentiment in valid_sentiments:
                if valid_sentiment in sentiment:
                    sentiment = valid_sentiment
                    break
            else:
                sentiment = "unknown (invalid output)"

        print(f"\nVietnamese Text: '{args.test_text_vi}'")
        print(f"Predicted Sentiment (AdaMergeX): '{sentiment}'")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters using AdaMergeX and evaluate for Sentiment Analysis.")
    
    # Paths to the three pre-trained adapters
    parser.add_argument("--adapter_l2_t1_path", required=True, help="Path to adapter: Target Task in Source Language (e.g., English Sentiment).")
    parser.add_argument("--adapter_name_l2_t1", required=True, help="Name used when loading adapter l2_t1 (e.g., 'en_sentiment'). Should match the name used during its PeftModel.from_pretrained or .add_adapter call.")
    
    parser.add_argument("--adapter_l1_t2_path", required=True, help="Path to adapter: Reference Task in Target Language (e.g., Vietnamese CLM).")
    parser.add_argument("--adapter_name_l1_t2", required=True, help="Name used when loading adapter l1_t2 (e.g., 'vi_clm').")

    parser.add_argument("--adapter_l2_t2_path", required=True, help="Path to adapter: Reference Task in Source Language (e.g., English CLM).")
    parser.add_argument("--adapter_name_l2_t2", required=True, help="Name used when loading adapter l2_t2 (e.g., 'en_clm').")

    # Merging and Model Config
    parser.add_argument("--base_model_name", type=str, default=BASE_MODEL_NAME, help="Base model Hugging Face ID.")
    parser.add_argument("--tokenizer_name", type=str, default=TOKENIZER_NAME, help="Tokenizer Hugging Face ID.")
    parser.add_argument("--lambda_hyper", type=float, default=DEFAULT_LAMBDA_HYPER, help="Lambda (τ) hyperparameter for merging.")
    parser.add_argument("--merged_adapter_save_path", type=str, default="./merged_vietnamese_sentiment_adamergex", help="Path to save the final merged adapter.")

    # Inference
    parser.add_argument("--test_text_vi", type=str, default="Bộ phim này rất tuyệt vời và đầy cảm xúc!", help="Sample Vietnamese text for sentiment inference after merging.")
    
    args = parser.parse_args()
    main(args)

