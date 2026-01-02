#!/usr/bin/env python
# coding=utf-8
"""Download and convert Alpaca dataset to chat SFT format.

Usage:
    python scripts/download_alpaca_sft.py
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm


def format_alpaca_to_chat(example):
    """
    Convert Alpaca format (instruction, input, output) to Chat SFT format.
    
    Alpaca data typically contains:
    - instruction: The instruction
    - input: Context input (may be empty)
    - output: The response
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output_text = example.get('output', '')

    # Build the User part content
    # If there is input, usually concatenate it after the instruction, or use it separately as context
    if input_text and input_text.strip():
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    # Build the data structure according to requirements
    chat_format = [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": output_text
        }
    ]
    
    return chat_format


def main():
    # 1. Set configuration
    # Recommend using 'yahma/alpaca-cleaned', a cleaned version of original Alpaca with hallucination and format errors fixed
    DATASET_NAME = "yahma/alpaca-cleaned"
    OUTPUT_DIR = "data"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "alpaca_cleaned_sft.jsonl")
    
    # Create data directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset: {DATASET_NAME} ...")
    
    # 2. Load data from Hugging Face
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"Download failed. Please check your network or Hugging Face connection. Error: {e}")
        return

    print(f"Dataset loaded successfully. Total {len(dataset)} entries. Converting format...")

    # 3. Convert and write to JSONL file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Use tqdm to display progress bar
        for example in tqdm(dataset):
            # Format the data
            formatted_entry = format_alpaca_to_chat(example)
            
            # Write one line of JSON
            f.write(json.dumps(formatted_entry, ensure_ascii=False) + '\n')

    print(f"\nConversion completed! File saved as: {OUTPUT_FILE}")
    print(f"Output sample (first line):")
    
    # Print the first line to see the result
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        print(f.readline())


if __name__ == "__main__":
    main()

