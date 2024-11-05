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

def load_true_answers(filename):
    """
    Load true answers from the dataset file.
    """
    true_answers = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            question = data['question']
            true_answers[question] = data['answers']
    return true_answers

# Load true answers from the provided file
true_answers_file = f'/home/student/ActiveRAG/data/data_{dataset}_sampled.jsonl'
true_answers = load_true_answers(true_answers_file)

def acc(output, answer_key):
    """
    Determine the correctness of the output based on the provided answer_key.
    Returns 'True' if any of the true answers match the output, otherwise 'False'.
    """
    if not answer_key:
        return 'False'  # No answers to compare against
    return 'True' if any(answer.lower() in output.lower() for answer in answer_key) else 'False'

# Path to save the CSV file
csv_file_path = f'vanila/{dataset}/top{topk}/vanilla_rag_results.csv'
csv_columns = ['id', 'vanilla_rag_output', 'correctness', 'true_answer']

# Create CSV file and write headers
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    # Loop through files and process each one
    for i in range(500):
        file_path = f'vanila/{dataset}/top{topk}/vanillarag/{dataset}_idx_{i}.json'

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # Extract the VanillaRAG output and question
                question = data.get("question", "")
                vanilla_rag_output = data.get("vanilla_rag_result", "")

                # Get the true answer for the question based on the question text
                answer_key = true_answers.get(question, [])

                # Calculate correctness using the provided answer_key
                correctness = acc(vanilla_rag_output, answer_key)

                # Write data to CSV
                csv_writer.writerow({
                    'id': i,  # You may use another field if available for ID
                    'vanilla_rag_output': vanilla_rag_output,
                    'correctness': correctness,
                    'true_answer': answer_key
                })
        else:
            print(f"File not found: {file_path}")

print("CSV file created successfully.")