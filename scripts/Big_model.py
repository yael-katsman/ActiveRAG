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

# Download NLTK data
nltk.download("stopwords")
nltk.download("punkt")

def calculate_bleu(reference, candidate):
    ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference]
    cand_tokens = nltk.word_tokenize(candidate.lower())
    smoothie = SmoothingFunction().method4
    return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smoothie)

def extract_agent_data(data_file_path, embedding_file_path):
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
    point_in_space = point_in_space.reshape(1, -1)
    similarities = {
        agent: cosine_similarity(embedding.reshape(1, -1), point_in_space)[0][0]
        for agent, embedding in agent_embeddings.items()
    }
    return max(similarities, key=similarities.get)

def calc_accuracy(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    correct_count = sum(1 for entry in data if entry['correctness'] is True)
    return (correct_count / len(data)) * 100 if data else 0

def calc_average_bleu_score(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return sum(entry['bleu_score'] for entry in data) / len(data) if data else 0

def get_agent_histogram(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    agent_counts = Counter(entry['chosen_agent'] for entry in data)
    return dict(agent_counts)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', type=str, required=True, help='Dataset to test on')
    parser.add_argument('--topk', type=int, required=True, help='Top K value (e.g., 5 or 10)')
    args = parser.parse_args()

    datasets = ['nq', 'popqa', 'triviaqa', 'webq']
    logs_dirs = [f"logs/{ds}/top{args.topk}" for ds in datasets]
    embeddings_dirs = [f"embeddings/{ds}/top{args.topk}" for ds in datasets]

    train_files, test_files = [], []
    for logs_dir, embeddings_dir in zip(logs_dirs, embeddings_dirs):
        data_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith('.json')]
        # Correct construction of embedding files from data files
        embedding_files = [
        os.path.join(embeddings_dir, os.path.basename(f).replace('.json', '_embeddings.json')) 
        for f in data_files
        ]


        split_idx = int(0.7 * len(data_files))
        train_files.extend(zip(data_files[:split_idx], embedding_files[:split_idx]))
        if args.test_dataset in logs_dir:
            test_files = list(zip(data_files[split_idx:], embedding_files[split_idx:]))

    train_data_files, train_embedding_files = zip(*train_files)
    test_data_files, test_embedding_files = zip(*test_files)

    train_loader = DataLoader(AgentDataset(train_data_files, train_embedding_files), batch_size=2, shuffle=True)
    test_loader = DataLoader(AgentDataset(test_data_files, test_embedding_files), batch_size=1, shuffle=False)

    model = AgentWeightingModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for embeddings, scores, agent_data, filename_tuple in train_loader:
            output, weights = model(embeddings, scores)
            target = embeddings.mean(dim=1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    all_test_results = []
    with torch.no_grad():
        model.eval()
        for embeddings, scores, agent_data, filename_tuple in test_loader:
            output, weights = model(embeddings, scores)
            closest_agent = find_closest_agent({a: np.array(data['embedding']) for a, data in agent_data.items()}, output[0].cpu().numpy())
            chosen_agent_correctness = agent_data[closest_agent]['correctness']
            result = {
                'filename': os.path.basename(filename_tuple[0]),
                'chosen_agent': closest_agent,
                'correctness': str(chosen_agent_correctness.item() if isinstance(chosen_agent_correctness, torch.Tensor) else chosen_agent_correctness),  # Convert tensor to scalar              
                'weights': weights.cpu().numpy().tolist(),
                'output': agent_data[closest_agent]['output'],
                'bleu_score': float(agent_data[closest_agent]['bleu_score'])
            }
            all_test_results.append(result)

    output_dir = f"Model_Answers/{args.test_dataset}/top{args.topk}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'results_big_model.json')

    with open(output_file, 'w') as f:
        json.dump(all_test_results, f, indent=4)

    accuracy = calc_accuracy(output_file)
    average_bleu = calc_average_bleu_score(output_file)
    agent_histogram = get_agent_histogram(output_file)

    summary = {
        'accuracy': f"{accuracy:.2f}%",
        'average_bleu_score': average_bleu,
        'agent_histogram': agent_histogram
    }

    summary_file = os.path.join(output_dir, 'summary_big_model.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Summary metrics saved in: {summary_file}")
