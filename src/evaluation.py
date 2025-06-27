from sentence_transformers import SentenceTransformer, util
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(llm_description, reference_description):
    """Evaluate the similarity between generated and reference descriptions."""
    llm_embedding = model.encode(llm_description, convert_to_tensor=True)
    ref_embedding = model.encode(reference_description, convert_to_tensor=True)
    return util.pytorch_cos_sim(llm_embedding, ref_embedding).item()


def compare_descriptions(llm_jsonl_path, reference_jsonl_path):
    """
    Compare descriptions from two JSONL files based on matching 'Filename' keys.
    Each JSONL file should be a list of dicts with 'Filename' and 'Description' keys.
    """
    with open(llm_jsonl_path, 'r') as f:
        llm_data = [json.loads(line) for line in f if line.strip()]
    with open(reference_jsonl_path, 'r') as f:
        ref_data = [json.loads(line) for line in f if line.strip()]

    # For the same filename in both datasets, calculate the similarity score on their descriptions
    results = []
    for llm_item in llm_data:
        llm_filename = llm_item.get('Filename')
        llm_description = llm_item.get('llm_generated_explanation', '')

        # Find the matching reference item
        ref_item = next((item for item in ref_data if item.get('Filename') == llm_filename), None)
        if ref_item:
            ref_description = ref_item.get('Description', '')
            similarity_score = calculate_similarity(llm_description, ref_description)
            results.append({
                'Filename': llm_filename,
                'ContrastElement': llm_item.get('ContrastElement', ''),
                'LLM_Description': llm_description,
                'Reference_Description': ref_description,
                'Similarity_Score': similarity_score
            })
    return results