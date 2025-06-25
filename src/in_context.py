import json
import random
import requests
import os

# --- Configuration ---
OFIQ_RESULTS_FILE = "data/QFIQ_assessment/tono_onot_assessment_combined.jsonl"
ICL_EXAMPLES_FILE = "data/QFIQ_assessment/incontext_test_data.jsonl" 
OUTPUT_LLM_EXPLANATIONS_FILE = "llm_generated_explanations.jsonl" # Output for LLM results

OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "llama3.2" # The name of the model you pulled with Ollama

# --- LLM System Prompt and Instruction (Common across all calls) ---
SYSTEM_PROMPT = """You are an AI assistant specialized in explaining face image quality defects for biometric compliance, based on OFIQ scores. Your task is to analyze the provided OFIQ scores for a face image and clearly state if the image can be accepted as a biometric or not. If not, explain any defects that need to be fixed for compliance, using the relevant OFIQ terminology.

Your explanations must:
- State if the image is compliant or not compliant.
- Point out one defect that must be fixed for compliance, if not compliant.
- Action, suggesting what needs to be improved, if relevant.
- Be in a professional and helpful tone.
- Referencing OFIQ components by their names (e.g., illumination, sharpness, head pose, eyes open, utility score).
"""

def load_data(filepath):
    """Loads JSON Lines data from a file."""
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    return data

def create_icl_prompt_section(icl_examples):
    """Creates the in-context learning examples string for the prompt."""
    icl_section = ""
    for i, example in enumerate(icl_examples):
        icl_section += f"OFIQ Scores: {json.dumps(example['OFIQResults'])}\n"
        icl_section += f"Correct Description: {example['Description']}\n\n"
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
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

def main():
    print("Loading in-context learning examples...")
    icl_examples = load_data(ICL_EXAMPLES_FILE)
    if not icl_examples:
        print(f"Error: No ICL examples found in {ICL_EXAMPLES_FILE}. Please create this file with your curated examples.")
        print("Each line should be a JSON object like: {\"Filename\": \"...\", \"OFIQResults\": {...}, \"correct_description\": \"...\"}")
        return

    print("Loading all OFIQ results for test set selection...")
    all_ofiqa_results = random.sample(load_data(OFIQ_RESULTS_FILE), 10)  # Load a sample of 1000 results for testing

    # --- Prepare your Test Set ---
    # IMPORTANT: Filter out images used in ICL_EXAMPLES_FILE to create your test set.
    # The test set should contain images that the LLM has NOT seen in the examples.
    icl_Filenames = {example['Filename'] for example in icl_examples}
    test_set_images = [
        item for item in all_ofiqa_results if item['Filename'] not in icl_Filenames
    ]

    print(f"Loaded {len(icl_examples)} ICL examples.")
    print(f"Identified {len(test_set_images)} images for the test set (excluding ICL examples).")

    if not test_set_images:
        print("Warning: Test set is empty. No new images to generate explanations for.")
        return

    # Create the ICL examples section for the prompt once
    icl_prompt_section = create_icl_prompt_section(icl_examples)

    print(f"Starting LLM explanation generation for {len(test_set_images)} images...")

    generated_explanations = []
    with open(OUTPUT_LLM_EXPLANATIONS_FILE, 'w') as outfile:
        for i, item in enumerate(test_set_images):
            Filename = item["Filename"]
            OFIQResults = item["OFIQResults"]
            ContrastElement = item['ContrastElement'] if 'ContrastElement' in item else "none"

            # Construct the full prompt for the current image
            current_image_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"{icl_prompt_section}"
                f"Now, explain if the biometrics can be accepted, if not then explain the primary defect for this image and provide actions to be taken:\n"
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
    main()