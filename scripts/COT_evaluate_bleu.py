import csv
import numpy as np
from argparse import ArgumentParser

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--topk', type=int, required=True)
args = parser.parse_args()

dataset = args.dataset
topk = args.topk

def evaluate(file_name):
    """
    Evaluate the average BLEU score of Chain of Thought (CoT) output based on the BLEU score column in the CSV file.
    """
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Calculate average BLEU score for the CoT output
    avg_bleu_score = calculate_average_bleu(data, 'bleu_score')
    print(f"{file_name} Average Chain of Thought (CoT) BLEU Score: {avg_bleu_score:.2f}%")

def calculate_average_bleu(data, column):
    """
    Calculate the average BLEU score of predictions based on the BLEU score column.
    """
    # Filter out any rows where the BLEU score is missing or cannot be converted to float
    bleu_scores = [float(row[column]) for row in data if row[column]]
    return np.mean(bleu_scores) * 100  # Convert to percentage

# Path to the CSV file containing CoT results
file_name = f'cot/{dataset}/top{topk}/cot_results_bleu.csv'

# Evaluate the results
evaluate(file_name)
