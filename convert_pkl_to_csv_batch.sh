#!/bin/bash

# Batch conversion script for PKL to CSV files
# Usage: ./batch_convert_pkl_to_csv.sh [input_dir] [output_dir]

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gmr

# Default paths
INPUT_DIR="${1:-/hhd4/lizhe/stcomp/robot_data/AMASS}"
OUTPUT_DIR="${2:-./converted_csv}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find all .pkl files and convert them
echo "Starting batch conversion..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Counter for processed files
count=0

# Process each PKL file
find "$INPUT_DIR" -name "*.pkl" -type f | while read -r pkl_file; do
    # 1. Extract the filename from the absolute path
    # Turns "/path/to/dance_sh_1.pkl.pkl" -> "dance_sh_1.pkl.pkl"
    filename=$(basename "$pkl_file")

    # 2. Eliminate content after the first "."
    # Turns "dance_sh_1.pkl.pkl" -> "dance_sh_1"
    clean_name="${filename%%.*}"

    # 3. Create the output path
    output_path="$OUTPUT_DIR/${clean_name}.csv"
    
    output_dir="$(dirname "$output_path")"
    
    # Create output subdirectory if needed
    mkdir -p "$output_dir"
    
    # Convert the file
    echo "Converting: $pkl_file"
    python ./scripts/convert_pkl_to_csv.py "$pkl_file" -o "$output_path"
    
    ((count++))
done

echo ""
echo "Batch conversion complete!"
echo "Total files processed: $count"
