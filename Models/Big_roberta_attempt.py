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
    def __init__(self, embedding_dim=768, num_agents=5, hidden_sizes=[128, 64]):
        super(AgentWeightingModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        layers = []
        input_dim = embedding_dim + 1
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

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
    correct_count = sum(1 for entry in data if entry['correctness'] == "True")
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
    datasets = ['nq', 'popqa', 'triviaqa', 'webq']
    top_k_values = [5, 10]
    logs_dirs = {ds: {k: f"logs/{ds}/top{k}" for k in top_k_values} for ds in datasets}
    embeddings_dirs = {ds: {k: f"Roberta_embeddings/{ds}/top{k}" for k in top_k_values} for ds in datasets}

    for k in top_k_values:
        # Gather all train files across datasets for the specific top_k value
        train_files = []
        for ds in datasets:
            data_files = [os.path.join(logs_dirs[ds][k], f) for f in os.listdir(logs_dirs[ds][k]) if f.endswith('.json')]
            embedding_files = [os.path.join(embeddings_dirs[ds][k], os.path.basename(f).replace('.json', '_embeddings.json')) for f in data_files]
            split_idx = int(0.7 * len(data_files))
            train_files.extend(zip(data_files[:split_idx], embedding_files[:split_idx]))

        train_data_files, train_embedding_files = zip(*train_files)
        train_loader = DataLoader(AgentDataset(train_data_files, train_embedding_files), batch_size=2, shuffle=True)
        learning_rate = 0.001
        model = AgentWeightingModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        epochs = 15

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for embeddings, scores, agent_data, filename_tuple in train_loader:
                output, weights = model(embeddings, scores)
                target = embeddings.mean(dim=1)
                loss = criterion(output, target)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

        # Testing only on the remaining 30% of top_k data for each dataset
        for ds in datasets:
            print(f"Testing on dataset: {ds}, top-K: {k}")
            test_files = []
            data_files = [os.path.join(logs_dirs[ds][k], f) for f in os.listdir(logs_dirs[ds][k]) if f.endswith('.json')]
            embedding_files = [os.path.join(embeddings_dirs[ds][k], os.path.basename(f).replace('.json', '_embeddings.json')) for f in data_files]
            split_idx = int(0.7 * len(data_files))
            test_files.extend(zip(data_files[split_idx:], embedding_files[split_idx:]))

            test_data_files, test_embedding_files = zip(*test_files)
            test_loader = DataLoader(AgentDataset(test_data_files, test_embedding_files), batch_size=1, shuffle=False)

            all_test_results = []
            with torch.no_grad():
                model.eval()
                for embeddings, scores, agent_data, filename_tuple in test_loader:
                    output, weights = model(embeddings, scores)
                    closest_agent = find_closest_agent(
                        {a: np.array(data['embedding']) for a, data in agent_data.items()},
                        output[0].cpu().numpy()
                    )
                    chosen_agent_correctness = agent_data[closest_agent]['correctness']
                    result = {
                        'filename': os.path.basename(filename_tuple[0]),
                        'chosen_agent': closest_agent,
                        'correctness': str(chosen_agent_correctness.item() if isinstance(chosen_agent_correctness, torch.Tensor) else chosen_agent_correctness),
                        'weights': weights.cpu().numpy().tolist(),
                        'output': agent_data[closest_agent]['output'],
                        'bleu_score': float(agent_data[closest_agent]['bleu_score'])
                    }
                    all_test_results.append(result)

            output_dir = f"Model_Answers/{ds}/top{k}"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'results_big_roberta.json')

            with open(output_file, 'w') as f:
                json.dump(all_test_results, f, indent=4)

            accuracy = calc_accuracy(output_file)
            average_bleu = calc_average_bleu_score(output_file)
            agent_histogram = get_agent_histogram(output_file)

            summary = {
                'accuracy': f"{accuracy:.2f}%",
                'average_bleu_score': average_bleu,
                'agent_histogram': agent_histogram,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'loss_function': criterion.__class__.__name__,
                'hidden_sizes': model.hidden_sizes,
                'optimizer': optimizer.__class__.__name__
            }

            summary_file = os.path.join(output_dir, 'summary_big_Roberta.json')
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

