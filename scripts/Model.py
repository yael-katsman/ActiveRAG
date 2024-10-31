import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("punkt")

epochs = 7
learning_rate = 0.001
datasets = ["triviaqa", "popqa", "nq", "webq"]
topk_values = [5, 10]  # Iterate over these values instead of passing them as arguments

def calculate_bleu(reference, candidate):
    """Calculate BLEU score between reference and candidate texts."""
    ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference]
    cand_tokens = nltk.word_tokenize(candidate.lower())
    smoothie = SmoothingFunction().method4
    return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smoothie)

def extract_agent_data(data_file_path, embedding_file_path):
    """Extract agent outputs and embeddings."""
    with open(data_file_path, 'r') as data_file:
        data = json.load(data_file)

    with open(embedding_file_path, 'r') as embed_file:
        embeddings_data = json.load(embed_file)

    agent_keys = ['cot', 'anchoring', 'associate', 'logician', 'cognition']
    true_answer = data['true_answer']
    agent_data = {}

    for key in agent_keys:
        if key in data and key in embeddings_data:
            output = data[key].get(f"{key}_output", "")
            embedding = embeddings_data[key]
            bleu_score = calculate_bleu(true_answer, output)

            correctness_key = f"{key}_correctness"
            agent_correctness = data.get(key, {}).get(correctness_key, "False").strip().lower() == "true"

            agent_data[key] = {
                'embedding': embedding,
                'bleu_score': bleu_score,
                'output': output,
                'correctness': agent_correctness
            }

    return agent_data

class AgentDataset(Dataset):
    """Custom dataset for loading data and embeddings."""
    def __init__(self, data_files, embedding_files):
        self.data = [
            extract_agent_data(data_file, embedding_file)
            for data_file, embedding_file in zip(data_files, embedding_files)
        ]
        self.filenames = data_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        agent_data = self.data[idx]
        embeddings = np.array([agent_data[a]['embedding'] for a in agent_data])
        bleu_scores = np.array([[agent_data[a]['bleu_score']] for a in agent_data])
        return (
            torch.tensor(embeddings, dtype=torch.float32),
            torch.tensor(bleu_scores, dtype=torch.float32),
            agent_data,
            self.filenames[idx]
        )

class AgentWeightingModel(nn.Module):
    """Neural network for learning agent weights."""
    def __init__(self, embedding_dim=768, num_agents=5):
        super(AgentWeightingModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, agent_embeddings, bleu_scores):
        inputs = torch.cat((agent_embeddings, bleu_scores), dim=-1)
        weights = self.fc(inputs).squeeze(-1)
        weighted_embeddings = weights.unsqueeze(-1) * agent_embeddings
        return weighted_embeddings.sum(dim=1), weights

def find_closest_agent(agent_embeddings, point_in_space):
    """Find the closest agent using cosine similarity."""
    point_in_space = point_in_space.reshape(1, -1)
    similarities = {
        agent: cosine_similarity(embedding.reshape(1, -1), point_in_space)[0][0]
        for agent, embedding in agent_embeddings.items()
    }
    return max(similarities, key=similarities.get)

def calc_accuracy(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    correct_count = sum(1 for entry in data if entry['correctness'] == "True")
    total_count = len(data)

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    return accuracy

def calc_average_bleu_score(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    total_bleu_score = sum(entry['bleu_score'] for entry in data)
    total_count = len(data)

    return total_bleu_score / total_count if total_count > 0 else 0

def get_agent_histogram(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    agent_counts = Counter(entry['chosen_agent'] for entry in data)
    return dict(agent_counts)

# Iterate over datasets and topk values
for dataset in datasets:
    for topk in topk_values:
        logs_dir = f"logs/{dataset}/top{topk}"
        embeddings_dir = f"embeddings/{dataset}/top{topk}"

        data_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith('.json')]
        embedding_files = [
            os.path.join(embeddings_dir, f"{os.path.splitext(f)[0]}_embeddings.json")
            for f in os.listdir(logs_dir) if f.endswith('.json')
        ]

        assert len(data_files) == len(embedding_files), "Mismatch between data and embedding files."

        train_size = int(0.7 * len(data_files))
        train_data, test_data = random_split(
            list(zip(data_files, embedding_files)),
            [train_size, len(data_files) - train_size]
        )

        train_data_files, train_embedding_files = zip(*train_data)
        test_data_files, test_embedding_files = zip(*test_data)

        train_loader = DataLoader(AgentDataset(train_data_files, train_embedding_files), batch_size=2, shuffle=True)
        test_loader = DataLoader(AgentDataset(test_data_files, test_embedding_files), batch_size=1, shuffle=False)

        model = AgentWeightingModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            model.train()
            for embeddings, scores, agent_data, filename_tuple in train_loader:
                filename = filename_tuple[0]
                output, weights = model(embeddings, scores)
                target = embeddings.mean(dim=1)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Dataset: {dataset}, TopK: {topk}, Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        # Testing loop
        model.eval()
        all_test_results = []
        with torch.no_grad():
            for embeddings, scores, agent_data, filename_tuple in test_loader:
                filename = filename_tuple[0]
                output, weights = model(embeddings, scores)

                agent_embeddings = {a: np.array(data['embedding']) for a, data in agent_data.items()}
                closest_agent = find_closest_agent(agent_embeddings, output[0].cpu().numpy())
                chosen_agent_correctness = agent_data[closest_agent]['correctness']

                result = {
                    'filename': os.path.basename(filename),
                    'chosen_agent': closest_agent,
                    'output': agent_data[closest_agent]['output'],
                    'bleu_score': float(agent_data[closest_agent]['bleu_score']),
                    'correctness': str(chosen_agent_correctness.item() if isinstance(chosen_agent_correctness, torch.Tensor) else chosen_agent_correctness),  # Convert tensor to scalar,
                    'weights': weights.cpu().numpy().tolist()
                }
                all_test_results.append(result)

        output_dir = f"Model_Answers/{dataset}/top{topk}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'results.json')

        with open(output_file, 'w') as f:
            json.dump(all_test_results, f, indent=4)

        accuracy = calc_accuracy(output_file)
        average_bleu_score = calc_average_bleu_score(output_file)
        agent_histogram = get_agent_histogram(output_file)

        summary = {
            'accuracy': f"{accuracy:.2f}%",
            'average_bleu_score': average_bleu_score,
            'agent_histogram': agent_histogram,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'loss_function': "CrossEntropyLoss"
        }

        summary_file = os.path.join(output_dir, 'summary_try.json')

        # Load existing summary data or start with an empty list
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                existing_summary = json.load(f)
                if isinstance(existing_summary, dict):
                    existing_summary = [existing_summary]
        except (FileNotFoundError, json.JSONDecodeError):
            existing_summary = []

        existing_summary.append(summary)

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(existing_summary, f, indent=4)

        print(f"Summary metrics saved in: {summary_file}")
