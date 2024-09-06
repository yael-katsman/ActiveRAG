import json
import os
import csv
from argparse import ArgumentParser

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--topk', type=int, required=True)
args = parser.parse_args()

dataset = args.dataset
topk = args.topk

def acc(output, answer_key):
    """
    Check if the output contains any of the correct answers.
    """
    return 'True' if any(answer.lower() in output.lower() for answer in answer_key) else 'False'

# Path to save the CSV file
csv_file_path = f'vanila/{dataset}/top{topk}/vanilla_rag_results.csv'
csv_columns = ['id', 'vanilla_rag_output', 'correctness', 'true_answer']

# Create CSV file and write headers
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    # Iterate over the expected number of files (or adjust as needed)
    for i in range(500):
        file_path = f'vanila/{dataset}/top{topk}/vanillarag/{dataset}_idx_{i}.json'

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # Extract the answer key and output
                answer_key = data.get("true_answer", [])
                vanilla_rag_output = data.get("vanilla_rag_result", "")

                # Calculate correctness
                correctness = acc(vanilla_rag_output, answer_key)

                # Write data to CSV
                csv_writer.writerow({
                    'id': i,
                    'vanilla_rag_output': vanilla_rag_output,
                    'correctness': correctness,
                    'true_answer': answer_key
                })
        else:
            print(f"File not found: {file_path}")

print("CSV file created successfully.")

# Usage example: python -m scripts.run_vanila --dataset nq --topk 5
