import json
import requests
import os

# --- Configuration ---
IN_CONTEXT_TEST_FILE = "data/QFIQ_assessment/in_context_test_data.jsonl"
IN_CONTEXT_TRAIN_FILE = "data/QFIQ_assessment/in_context_train1.jsonl" 
OUTPUT_LLM_EXPLANATIONS_FILE = "llm_generated_explanations_123.jsonl" # Output for LLM results
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "llama3.2"
SYSTEM_PROMPT = """You are an AI assistant specialized in explaining face image quality defects for biometric compliance, based on OFIQ scores. 
Your task is to analyze the provided OFIQ scores for a face image and clearly state for the person who took the photo if the image can be accepted as a biometric or not. 
If not, give the description of the defect that need to be fixed for compliance. There can be at most one main defect per image.

Your description must:
- State if the image is compliant or not compliant.
    If it is compliant, state that the image is fully compliant and no specific defects were found. Otherwise:
    - Point out one defect that is present in the picture that is not complaint.
    - Action, suggesting what to do to remove the defect.
- Use a professional and easy to uderstand tone.
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
                f"Now, describe if the following image is compliant.\n\n"
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



import json
import requests
import os

# --- Configuration ---
IN_CONTEXT_TEST_FILE = "data/QFIQ_assessment/in_context_test_data.jsonl"
IN_CONTEXT_TRAIN_FILE = "data/QFIQ_assessment/in_context_train1.jsonl" 
OUTPUT_LLM_EXPLANATIONS_FILE = "llm_generated_explanations_321.jsonl" # Output for LLM results
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "llama3.2"
SYSTEM_PROMPT = """You are an AI assistant specialized in explaining face image quality defects for biometric compliance, based on OFIQ scores. The scores are given in range from 0 to 100, where 100 indicates optimal quality, but in the HeadSize component the higher score might indicate a defect.
Your task is to analyze the provided OFIQ scores for a face image and clearly state for the person who took the photo if the image can be accepted as a biometric or not. 
If not, give the description of the defect that need to be fixed for compliance. There can be at most one main defect per image.

Your description must:
- State if the image is compliant or not compliant.
- Mention only the speccific OFIQ scores that indicate the defect (if relevant), but not the one that are in the normal values.
    If it is compliant, state that the image is fully compliant and no specific defects were found. Otherwise:
    - Point out one defect that is present in the picture that is not complaint.
    - Action, suggesting what to do to remove the defect.
- Use a professional and easy to uderstand tone.
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
                f"Now, explain the following image. Don't explain all the scores but only mention scores that clearly indicate found defect\n\n"
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


import json
import requests
import os

# --- Configuration ---
IN_CONTEXT_TEST_FILE = "data/QFIQ_assessment/in_context_test_data.jsonl"
IN_CONTEXT_TRAIN_FILE = "data/QFIQ_assessment/in_context_train1.jsonl" 
OUTPUT_LLM_EXPLANATIONS_FILE = "llm_generated_explanations_111.jsonl" # Output for LLM results
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "llama3.2"
SYSTEM_PROMPT = """You are an AI assistant specialized in explaining face image quality defects for biometric compliance, based on OFIQ scores.
Your task is to analyze the provided OFIQ scores for a face image and provide a precise, actionable assessment, strictly following the specified output format.
OFIQ scores range from 0 to 100, where 100 indicates optimal quality. For **HeadSize.scalar**, however, this value is inveres.

Your response MUST adhere to the following structure and content rules:

1.  **Overall Compliance Statement:** Start directly with "The image is compliant for biometric use." OR "The image is not compliant for biometric use." (exactly as written).
2.  **Defect Details (IF NOT COMPLIANT):**
    * If the image is "not compliant," immediately follow the compliance statement with a clear, concise description of the **primary underlying issue** (e.g. "improper zoom", "subject is wearing a cap")
    * Then, detail all **related OFIQ scalar scores** that deviates mostly from compliant/normal values and their specific impact. Explain how these scores manifest the primary issue. Do NOT mention scores that are within normal ranges.
    * Ensure the explanation links multiple related scalar scores if they stem from the same primary cause (e.g., improper zoom impacting HeadSize, InterEyeDistance, Sharpness, and Margins).
3.  **Actionable Feedback:** Conclude with a clear, concise sentence starting "Actionable Feedback: " followed by practical advice on what to do to rectify the *entire set of related issues* stemming from the primary defect.
4.  **IF COMPLIANT:** If the image is "compliant," after the compliance statement, simply state "No specific defects were found. This image meets all biometric requirements." Do NOT provide any defect details or actionable feedback.

Use professional and easy-to-understand language. Do not output anything else besides the requested explanation.
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
        # User's input for the ICL example
        icl_section += f"User:\n"
        icl_section += f"OFIQ Scores: {json.dumps(example['OFIQResults'])}\n"
        
        # Assistant's expected output for the ICL example
        icl_section += f"\nAssistant:\n"
        icl_section += f"{example['Description'].strip()}\n\n"
    
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
                f"{SYSTEM_PROMPT}\n\n" # This comes first and sets the rules
                f"{icl_prompt_section}" # Then the training examples, each with User/Assistant turns
                
                # The final User turn for the current image
                f"User:\n"
                f"OFIQ Scores: {json.dumps(OFIQResults)}\n"
                
                # The final Assistant turn, prompting the LLM to generate its response
                f"\nAssistant:" # Important: no newline after "Assistant:" if you want the LLM to start immediately
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

import json
import requests
import os

# --- Configuration ---
IN_CONTEXT_TEST_FILE = "data/QFIQ_assessment/in_context_test_data.jsonl"
IN_CONTEXT_TRAIN_FILE = "data/QFIQ_assessment/in_context_train1.jsonl" 
OUTPUT_LLM_EXPLANATIONS_FILE = "llm_generated_explanations_100.jsonl" # Output for LLM results
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "llama3.2"
SYSTEM_PROMPT = """
You are an AI assistant specializing in image compliance analysis. Your task is to evaluate images based on a provided set of OFIQ scores values representing their key components. For each image, you will determine if it is "Compliant" or "Not Compliant."

If an image is Not Compliant, you must identify the single reason for non-compliance and provide clear, actionable feedback on how to rectify the issue.

Output an analysis for each image in the following format:
- Compliance Status: [Compliant/Not Compliant]
- Issue: [Brief description of the non-compliance issue. This should only be present if the status is "Not Compliant".]
- Actionable Feedback: [Specific instructions on what to change to make the image compliant. This should only be present if the status is "Not Compliant".]

Here are some examples of the expected output:
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
        # User's input for the ICL example
        icl_section += f"OFIQ Scores: {json.dumps(example['OFIQResults'])}\n"
        icl_section += f"{example['Description'].strip()}\n\n"
    
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
                f"{SYSTEM_PROMPT}\n\n" # This comes first and sets the rules
                f"{icl_prompt_section}" # Then the training examples, each with User/Assistant turns
                
                # The final User turn for the current image
                f"Now, analyze the following image based on its scalar values:\n"
                f"OFIQ Scores: {json.dumps(OFIQResults)}\n"
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


import json
import requests
import os

# --- Configuration ---
IN_CONTEXT_TEST_FILE = "data/QFIQ_assessment/in_context_test_data.jsonl"
IN_CONTEXT_TRAIN_FILE = "data/QFIQ_assessment/in_context_train1.jsonl" 
OUTPUT_LLM_EXPLANATIONS_FILE = "llm_generated_explanations_new.jsonl" # Output for LLM results
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "llama3.2"
SYSTEM_PROMPT = """
You are an AI assistant specializing in image compliance analysis. Your task is to evaluate images based on their OFIQ scores and generate descriptions that exactly match the format of the examples.

Required Format:
For compliant images (use exactly this text):
"This image is fully compliant for biometric use and without any defects. No changes needed."

For non-compliant images (use exactly this structure):
"This image is not compliant for biometric use. The {metric} is {score}. This score indicates {brief explanation}. Actionable Feedback: {specific instructions}"

Important rules:
1. Always use the exact formats above - do not deviate or add extra text
2. For non-compliant images:
   - Focus on the single most significant quality issue
   - Include the specific metric name and score (e.g., "BackgroundUniformity.scalar: 40")
   - Provide clear, actionable feedback that directly addresses the issue
3. Be consistent in terminology:
   - Use "not compliant" for failing images (never "non-compliant" or "incompliant")
   - Use the exact metric names from the OFIQ scores
4. Keep explanations brief and focused on the single main issue
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
        # User's input for the ICL example
        icl_section += f"OFIQ Scores: {json.dumps(example['OFIQResults'])}\n"
        icl_section += f"Main Defect: {example['ContrastElement']} \n"
        icl_section += f"{example['Description'].strip()}\n\n"
    
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
                
                # The final User turn for the current image
                f"Now, describe the following image based on its QFIQ scores. Don't explain all the values, but only the relevant ones that caused the incompliance. Remember to provide actionable feedback:\n"
                f"OFIQ Scores: {json.dumps(OFIQResults)}\n"
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