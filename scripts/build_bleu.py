import json
import os
import csv
from argparse import ArgumentParser
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download necessary NLTK resources
nltk.download('punkt')

parser = ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--topk', type=int, required=True)
args = parser.parse_args()

dataset = args.dataset
topk = args.topk

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference]
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)

csv_file_path = f'log/{dataset}/top{topk}/prompt_bleu.csv'
csv_columns = ['id', 'anchoring_output', 'anchoring_bleu', 'associate_output', 'associate_bleu',
               'logician_output', 'logician_bleu', 'cognition_output', 'cognition_bleu', 'true_answer']

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    for i in range(500):
        file_path = f'log/{dataset}/top{topk}/{dataset}_idx_{i}.json'

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                answer_key = data["question_info"]["__answers__"]

                anchoring_output = data["anchoring"][-1]["content"]
                anchoring_bleu = calculate_bleu(answer_key, anchoring_output)

                associate_output = data["associate"][-1]["content"]
                associate_bleu = calculate_bleu(answer_key, associate_output)

                logician_output = data["logician"][-1]["content"]
                logician_bleu = calculate_bleu(answer_key, logician_output)

                cognition_output = data["cognition"][-1]["content"]
                cognition_bleu = calculate_bleu(answer_key, cognition_output)

                csv_writer.writerow(
                    {'id': i, 'anchoring_output': anchoring_output, 'anchoring_bleu': anchoring_bleu,
                     'associate_output': associate_output, 'associate_bleu': associate_bleu,
                     'logician_output': logician_output, 'logician_bleu': logician_bleu,
                     'cognition_output': cognition_output, 'cognition_bleu': cognition_bleu,
                     'true_answer': answer_key})
        else:
            print(f"File not found: {file_path}")

print("CSV file created successfully.")
