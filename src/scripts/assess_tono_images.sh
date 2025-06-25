#!/bin/bash
# Script to assess all images in data/TONO/release and write results to a CSV file using OFIQSampleApp

# Set paths
OFIQ_APP="../../OFIQ-Project/install_x86_64_linux/Release/bin/OFIQSampleApp"
CONFIG_DIR="../../OFIQ-Project/data"
IMAGE_DIR="../../src/data/TONO/release"
OUTPUT_CSV="../../src/data/QFIQ_assessment/tono_assessment.csv"
TMP_IMAGE_LIST="/tmp/tono_image_list.txt"

# Check if OFIQSampleApp exists
if [ ! -f "$OFIQ_APP" ]; then
  echo "Error: OFIQSampleApp not found at $OFIQ_APP"
  exit 1
fi

# Find all images in the directory (recursively)
find "$IMAGE_DIR" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) > "$TMP_IMAGE_LIST"

if [ ! -s "$TMP_IMAGE_LIST" ]; then
  echo "Error: No images found in $IMAGE_DIR"
  exit 1
fi

# Run the assessment
LD_LIBRARY_PATH="$(dirname $OFIQ_APP)" "$OFIQ_APP" -c "$CONFIG_DIR" -i "$TMP_IMAGE_LIST" -o "$OUTPUT_CSV"

if [ $? -eq 0 ]; then
  echo "Assessment complete. Results saved to $OUTPUT_CSV."
else
  echo "Assessment failed."
  exit 1
fi
