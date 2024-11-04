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
    Evaluate the accuracy of VanillaRAG output based on the correctness column in the CSV file.
    """
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Calculate accuracy for the VanillaRAG output
    column_accuracy = calculate_accuracy(data, 'correctness')
    print(f"{file_name} VanillaRAG Accuracy: {column_accuracy:.2f}%")

def calculate_accuracy(data, column):
    """
    Calculate the accuracy of predictions based on the correctness column.
    """
    correctness_values = [row[column] for row in data]
    accuracy = 100 * np.mean([1 if correctness == 'True' else 0 for correctness in correctness_values])
    return accuracy

# Path to the CSV file containing VanillaRAG results
file_name = f'vanila/{dataset}/top{topk}/vanilla_rag_results.csv'

# Evaluate the results
evaluate(file_name)
