import csv
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--topk', type=int, required=True)
args = parser.parse_args()

dataset = args.dataset
topk = args.topk

# Define BLEU score columns to evaluate
bleu_columns = ['anchoring_bleu', 'associate_bleu', 'logician_bleu', 'cognition_bleu']

def evaluate(file_name):
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    for column in bleu_columns:
        column_bleu = calculate_average_bleu(data, column)
        print(f"{file_name} {column} Average BLEU: {column_bleu:.2f}")

def calculate_average_bleu(data, column):
    bleu_scores = [float(row[column]) for row in data if row[column]]
    average_bleu = np.mean(bleu_scores) * 100  # Multiplying by 100 for percentage representation
    return average_bleu

# Update file path to match the BLEU-based CSV file
file_name = f'log/{dataset}/top{topk}/prompt_bleu.csv'

evaluate(file_name)
