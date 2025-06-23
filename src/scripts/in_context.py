import json
import requests
import os

# --- Configuration ---
OFIQ_RESULTS_FILE = "./incontext_test_data.json" # Your complete OFIQ assessment results file
ICL_EXAMPLES_FILE = "icl_examples.jsonl" # File where you saved your crafted ICL examples
OUTPUT_LLM_EXPLANATIONS_FILE = "llm_generated_explanations.jsonl" # Output for LLM results

OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "llama3" # The name of the model you pulled with Ollama

# --- LLM System Prompt and Instruction (Common across all calls) ---
SYSTEM_PROMPT = """You are an AI assistant specialized in explaining face image quality defects for biometric compliance, based on OFIQ scores. Your task is to analyze the provided OFIQ scores for a face image and clearly explain any defects that need to be fixed for compliance, using the relevant OFIQ terminology.

Your explanations should be:
- Concise but comprehensive.
- Focused on the most relevant defects to fix.
- Actionable, suggesting what needs to be improved.
- In a professional and helpful tone.
- Referencing OFIQ components by their names (e.g., illumination, sharpness, head pose, eyes open, utility score).
- State if the image is compliant or non-compliant overall, referencing the overall utility score.
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
        icl_section += f"--- Example {i+1}:\n"
        icl_section += f"OFIQ Scores: {json.dumps(example['ofiqa_scores'])}\n"
        icl_section += f"Correct Description: {example['correct_description']}\n\n"
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
            # "top_k": 40,
            # "top_p": 0.9,
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
        print("Each line should be a JSON object like: {\"image_path\": \"...\", \"ofiqa_scores\": {...}, \"correct_description\": \"...\"}")
        return

    print("Loading all OFIQ results for test set selection...")
    all_ofiqa_results = load_data(OFIQ_RESULTS_FILE)

    # --- Prepare your Test Set ---
    # IMPORTANT: Filter out images used in ICL_EXAMPLES_FILE to create your test set.
    # The test set should contain images that the LLM has NOT seen in the examples.
    icl_image_paths = {example['image_path'] for example in icl_examples}
    test_set_images = [
        item for item in all_ofiqa_results if item['image_path'] not in icl_image_paths
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
            image_path = item["image_path"]
            ofiqa_scores = item["ofiqa_scores"]

            # Construct the full prompt for the current image
            current_image_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"{icl_prompt_section}"
                f"Now, explain the defects for this image:\n"
                f"OFIQ Scores: {json.dumps(ofiqa_scores)}\n"
                f"Explanation:"
            )

            print(f"Processing image {i+1}/{len(test_set_images)}: {os.path.basename(image_path)}")
            llm_response = call_ollama(current_image_prompt)

            if llm_response:
                result_entry = {
                    "image_path": image_path,
                    "ofiqa_scores": ofiqa_scores,
                    "llm_generated_explanation": llm_response.strip()
                }
                generated_explanations.append(result_entry)
                outfile.write(json.dumps(result_entry) + '\n')
            else:
                print(f"Failed to get LLM response for {os.path.basename(image_path)}")

    print(f"\nLLM explanation generation complete. Results saved to {OUTPUT_LLM_EXPLANATIONS_FILE}")
    print(f"Successfully generated explanations for {len(generated_explanations)} images.")

if __name__ == "__main__":
    main()