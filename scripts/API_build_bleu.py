import json
import os
import csv
from argparse import ArgumentParser
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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

def calculate_bleu(output, reference):
    """
    Calculate the BLEU score for a single output against the reference answers using tokenization and smoothing.
    """
    # Tokenize the output and reference answers
    reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference]
    candidate_tokens = nltk.word_tokenize(output.lower())
    
    # Use a smoothing function to handle cases with no overlapping n-grams
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)

# Path to save the CSV file
csv_file_path = f'api_4_mini/{dataset}/results/api_results_bleu.csv'
csv_columns = ['id', 'api_output', 'bleu_score', 'true_answer']

# Create CSV file and write headers
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    # Loop through files and process each one
    for i in range(500):  # Adjust 500 based on the number of data points
        file_path = f'api_4_mini/{dataset}/results/{dataset}_idx_{i}.json'

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # Extract the API output and question
                question = data.get("question", "")
                api_output = data.get("api_result", "")

                # Get the true answer for the question based on the question text
                answer_key = true_answers.get(question, [])

                # Calculate BLEU score using the provided answer_key
                bleu_score = calculate_bleu(api_output, answer_key)

                # Write data to CSV
                csv_writer.writerow({
                    'id': i,  # You may use another field if available for ID
                    'api_output': api_output,
                    'bleu_score': bleu_score,
                    'true_answer': answer_key
                })
        else:
            print(f"File not found: {file_path}")

print("CSV file created successfully.")
