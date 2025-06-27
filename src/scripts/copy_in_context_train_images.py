import os
import json
import shutil

# Path to the in-context train JSONL file
IN_CONTEXT_TRAIN_FILE = 'src/data/QFIQ_assessment/in_context_train_data.jsonl'
# Root folders for TONO and ONOT images
TONO_ROOT = 'src/data/TONO/release/'
ONOT_ROOT = 'src/data/ONOT/S115379105/'
# Folder to copy images to
OUTPUT_FOLDER = 'src/data/QFIQ_assessment/in_context_train_images/'

with open(IN_CONTEXT_TRAIN_FILE, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        filename = item['Filename']
        contrast_element = item.get('ContrastElement')
        # Determine source path and output subfolder
        if contrast_element and contrast_element != 'ONOT':
            # TONO image: subfolder is the contrast element
            src_path = os.path.join(TONO_ROOT, contrast_element, os.path.basename(filename))
            out_subfolder = os.path.join(OUTPUT_FOLDER, contrast_element)
        else:
            # ONOT image
            src_path = os.path.join(ONOT_ROOT, os.path.basename(filename))
            out_subfolder = os.path.join(OUTPUT_FOLDER, 'ONOT')
        os.makedirs(out_subfolder, exist_ok=True)
        dst_path = os.path.join(out_subfolder, os.path.basename(filename))
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} -> {dst_path}")
        else:
            print(f"WARNING: Source image not found: {src_path}")
