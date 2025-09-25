import argparse
import os
from nlp_pipeline.processor import process_text_file

def process_folder(input_dir, output_dir):
    """
    Processes all .txt files in the specified input directory and saves outputs to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            process_text_file(filepath, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the linguistic pipeline on a folder of text files.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input folder containing .txt files.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder for results.")

    args = parser.parse_args()
    process_folder(args.input, args.output)