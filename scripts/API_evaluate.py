import csv
import numpy as np
from argparse import ArgumentParser

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

dataset = args.dataset

def evaluate(file_name):
    """
    Evaluate the accuracy of API output based on the correctness column in the CSV file.
    """
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Calculate accuracy for the API output
    column_accuracy = calculate_accuracy(data, 'correctness')
    print(f"{file_name} API Accuracy: {column_accuracy:.2f}%")

def calculate_accuracy(data, column):
    """
    Calculate the accuracy of predictions based on the correctness column.
    """
    correctness_values = [row[column] for row in data]
    accuracy = 100 * np.mean([1 if correctness == 'True' else 0 for correctness in correctness_values])
    return accuracy

# Path to the CSV file containing API results
file_name = f'api_4_mini/{dataset}/results/api_results.csv'

# Evaluate the results
evaluate(file_name)
