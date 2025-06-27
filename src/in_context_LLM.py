import json
import random
import requests
import os

# --- Configuration ---
IN_CONTEXT_TEST_FILE = "data/QFIQ_assessment/in_context_test_data.jsonl"
IN_CONTEXT_TRAIN_FILE = "data/QFIQ_assessment/in_context_train_data.jsonl" 
OUTPUT_LLM_EXPLANATIONS_FILE = "data/llm_descriptions/llm_generated_explanations.jsonl" # Output for LLM results
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "llama3.2"
SYSTEM_PROMPT = """You are an AI assistant specialized in explaining face image quality defects for biometric compliance, based on OFIQ ( Open Source Face Image Quality) assessment.
 Your primary task is to identify and clearly explain the single most significant defect detected in a face image, based on its OFIQ scalar scores. There might be images that don't have any defects, in which case you should state that the image is compliant for biometric use.
   Your explanation must always reference the relevant OFIQ scalar values and provide clear, actionable feedback for change.
   Always start with stating if the image is compliant or not compliant. If not, state the main defect by referencing the contrast element.
    Note that the scalar range is from 0 to 100. While higher scores generally indicate better quality, some metrics may have an optimal range or an inverse interpretation for extreme values (e.g., 0 or 100) ex. HeadSize.scalar.
    Your response should follow the structure demonstrated in the examples, focusing on one primary defect per image explanation.
"""

def load_data(filepath):
    """Loads JSON Lines data from a file."""
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    return data

def create_icl_prompt_section(in_context_train):
    """Creates the in-context learning examples string for the prompt."""
    icl_section = ""
    for i, example in enumerate(in_context_train):
        icl_section += f"OFIQ Scores: {json.dumps(example['OFIQResults'])}\n"
        icl_section += f"Correct Description: {example['Description']}\n\n"
        icl_section += f"Key Defect: {example['ContrastElement']}\n"
    return icl_section

def call_ollama(prompt):
    """Calls the Ollama API to generate a response."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False, # Set to True for streaming response
        "options": {
            "temperature": 0.1, # Lower temperature for less creativity, more deterministic output
            "top_k": 40,
            "top_p": 0.9,
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

def generate_LLM_descriptions():
    print("Loading in-context learning examples...")
    in_context_train = load_data(IN_CONTEXT_TRAIN_FILE)
    if not in_context_train:
        print(f"Error: No ICL examples found in {IN_CONTEXT_TRAIN_FILE}. Please create this file with your curated examples.")
        return
    print(f"Loaded {len(in_context_train)} train examples.")
    
    print("Loading test data...")
    test_set_images = load_data(IN_CONTEXT_TEST_FILE)
    print(f"Identified {len(test_set_images)} images for the test set.")


    icl_prompt_section = create_icl_prompt_section(in_context_train)

    print(f"Starting LLM explanation generation for {len(test_set_images)} images...")
    generated_explanations = []
    with open(OUTPUT_LLM_EXPLANATIONS_FILE, 'w') as outfile:
        for i, item in enumerate(test_set_images):
            Filename = item["Filename"]
            OFIQResults = item["OFIQResults"]
            ContrastElement = item['ContrastElement']

            # Construct the full prompt for the current image
            current_image_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"{icl_prompt_section}"
                f"Now, explain if the biometrics is compliant. Your response should always start with an overall compliance assessment. Then, detail any specific defects, referencing the relevant OFIQ scalar scores. Conclude with clear, actionable recommendations for improvement. Keep it short and concise\n\n"
                f"Filename: {Filename}\n"
                f"OFIQ Scores: {json.dumps(OFIQResults)}\n"
                f"Explanation:"
            )

            print(f"Processing image {i+1}/{len(test_set_images)}: {os.path.basename(Filename)}")
            llm_response = call_ollama(current_image_prompt)

            if llm_response:
                result_entry = {
                    "Filename": Filename,
                    "ContrastElement": ContrastElement,
                    "OFIQResults": OFIQResults,
                    "llm_generated_explanation": llm_response.strip()
                }
                generated_explanations.append(result_entry)
                outfile.write(json.dumps(result_entry) + '\n')
            else:
                print(f"Failed to get LLM response for {os.path.basename(Filename)}")

    print(f"\nLLM explanation generation complete. Results saved to {OUTPUT_LLM_EXPLANATIONS_FILE}")
    print(f"Successfully generated explanations for {len(generated_explanations)} images.")

if __name__ == "__main__":
    generate_LLM_descriptions()