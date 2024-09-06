import json
import os
import csv
from argparse import ArgumentParser

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

dataset = args.dataset

def load_true_answers(filename):
    """
    Load the true answers from the dataset file.
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
    """
    if not answer_key:
        return 'False'  # No answers to compare against
    return 'True' if any(answer.lower() in output.lower() for answer in answer_key) else 'False'

# Path to save the CSV file with results
csv_file_path = f'api_4/{dataset}/results/api_results.csv'
csv_columns = ['id', 'api_output', 'correctness', 'true_answer']

# Create CSV file and write headers
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    # Loop through files and process each one
    for i in range(500):  # Adjust 500 to your actual number of data points
        file_path = f'api_4/{dataset}/results/{dataset}_idx_{i}.json'

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # Extract the API output and question
                question = data.get("question", "")
                api_output = data.get("api_result", "")

                # Get the true answer for the question based on the question text
                answer_key = true_answers.get(question, [])

                # Calculate correctness using the provided answer_key
                correctness = acc(api_output, answer_key)

                # Write data to CSV
                csv_writer.writerow({
                    'id': i,  # You may use another field if available for ID
                    'api_output': api_output,
                    'correctness': correctness,
                    'true_answer': answer_key
                })
        else:
            print(f"File not found: {file_path}")

print("CSV file created successfully.")
