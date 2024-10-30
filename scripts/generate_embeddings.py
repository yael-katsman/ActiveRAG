import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("punkt")

# Initialize BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Set stop words
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """Preprocess text by removing non-alphabetic characters and stopwords."""
    words = re.findall(r'\b\w+\b', text.lower())
    return ' '.join([word for word in words if word not in stop_words])

def get_embedding(text):
    """Generate a 768-dimensional BERT embedding for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()  # Convert to list

def save_embeddings_for_file(input_path, output_path):
    """Generate and save embeddings for a single JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    embeddings = {}
    for key in ['cot', 'anchoring', 'associate', 'logician', 'cognition']:
        if key in data:
            output = preprocess_text(data[key].get(f"{key}_output", ""))
            embedding = get_embedding(output)
            embeddings[key] = embedding

    with open(output_path, 'w') as out_file:
        json.dump(embeddings, out_file, indent=4)

    print(f"Saved embeddings to {output_path}")

def process_dataset_and_topk(dataset_dir):
    """Process all top-k folders and JSON files within each dataset."""
    for topk_folder in os.listdir(dataset_dir):
        topk_path = os.path.join(dataset_dir, topk_folder)
        if os.path.isdir(topk_path):  # Ensure it's a directory
            print(f"Processing: {topk_path}")
            process_json_files_in_directory(topk_path)

def process_json_files_in_directory(directory):
    """Generate embeddings for all JSON files in the given directory."""
    for file_name in os.listdir(directory):
        if file_name.endswith('.json'):
            input_path = os.path.join(directory, file_name)
            output_dir = directory.replace('logs', 'embeddings')
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, file_name.replace('.json', '_embeddings.json'))
            if os.path.exists(output_path):
                print(f"Embeddings already exist for {file_name}. Skipping...")
            else:
                save_embeddings_for_file(input_path, output_path)

if __name__ == "__main__":
    # Define the root logs directory containing datasets
    logs_dir = "logs"  # Example: logs/nq/top10

    if not os.path.exists(logs_dir):
        print(f"Logs directory '{logs_dir}' not found!")
        exit(1)

    # Process each dataset directory (nq, popqa, triviaqa, webq, etc.)
    for dataset in os.listdir(logs_dir):
        dataset_path = os.path.join(logs_dir, dataset)
        if os.path.isdir(dataset_path):  # Ensure it's a directory
            print(f"Processing dataset: {dataset}")
            process_dataset_and_topk(dataset_path)

    print("Embedding generation completed for all datasets.")
