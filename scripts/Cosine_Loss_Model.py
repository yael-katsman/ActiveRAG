import json
import os
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
            agent_correctness = str(data.get(key, {}).get(correctness_key, "False")[0])

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
        embeddings = np.array([agent_data[a]['embedding'] for a in agent_data], dtype=object)
        bleu_scores = np.array([[agent_data[a]['bleu_score']] for a in agent_data])
        return (
            torch.tensor(np.stack(embeddings), dtype=torch.float32),
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

def calc_accuracy(results):
    correct_count = sum(1 for entry in results if entry['correctness'] == "True")
    total_count = len(results)
    return (correct_count / total_count) * 100 if total_count > 0 else 0

def calc_average_bleu_score(results):
    total_bleu_score = sum(entry['bleu_score'] for entry in results)
    return total_bleu_score / len(results) if results else 0

def get_agent_histogram(results):
    agent_counts = Counter(entry['chosen_agent'] for entry in results)
    return dict(agent_counts)

# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., triviaqa)')
    parser.add_argument('--topk', type=int, required=True, help='Top K value (e.g., 10)')
    args = parser.parse_args()

    logs_dir = f"logs/{args.dataset}/top{args.topk}"
    embeddings_dir = f"embeddings/{args.dataset}/top{args.topk}"

    data_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith('.json')]
    embedding_files = [os.path.join(embeddings_dir, f"{os.path.splitext(f)[0]}_embeddings.json") for f in data_files]

    assert len(data_files) == len(embedding_files), "Mismatch between data and embedding files."

    train_size = int(0.7 * len(data_files))
    train_data, test_data = random_split(list(zip(data_files, embedding_files)), [train_size, len(data_files) - train_size])

    train_data_files, train_embedding_files = zip(*train_data)
    test_data_files, test_embedding_files = zip(*test_data)

    train_loader = DataLoader(AgentDataset(train_data_files, train_embedding_files), batch_size=2, shuffle=True)
    test_loader = DataLoader(AgentDataset(test_data_files, test_embedding_files), batch_size=1, shuffle=False)

    model = AgentWeightingModel()
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        model.train()
        for embeddings, scores, agent_data, filename_tuple in train_loader:
            filename = filename_tuple[0]
            output, weights = model(embeddings, scores)
            target = embeddings.mean(dim=1)  
            loss = criterion(output, target, torch.ones(output.size(0)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")

    # Testing loop
    model.eval()
    all_test_results = []
    with torch.no_grad():
        for embeddings, scores, agent_data, filename_tuple in test_loader:
            filename = filename_tuple[0]
            output, weights = model(embeddings, scores)

            agent_embeddings = {a: np.array(data['embedding'], dtype=float) for a, data in agent_data.items()}
            closest_agent = find_closest_agent(agent_embeddings, output[0].cpu().numpy())

            result = {
                'filename': os.path.basename(filename),
                'chosen_agent': closest_agent,
                'output': agent_data[closest_agent]['output'],
                'bleu_score': float(agent_data[closest_agent]['bleu_score']),
                'correctness': agent_data[closest_agent]['correctness'],
                'weights': weights.cpu().numpy().tolist()
            }
            all_test_results.append(result)

    output_dir = f"Model_Answers/{args.dataset}/top{args.topk}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'results.json')

    with open(output_file, 'w') as f:
        json.dump(all_test_results, f, indent=4)

    print(f"All test results saved in: {output_file}")

    accuracy = calc_accuracy(all_test_results)
    avg_bleu = calc_average_bleu_score(all_test_results)
    histogram = get_agent_histogram(all_test_results)

    summary = {
        "accuracy": f"{accuracy:.2f}%",
        "average_bleu_score": avg_bleu,
        "agent_histogram": histogram
    }

    summary_file = os.path.join(output_dir, 'summary_cosine.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Summary metrics saved in: {summary_file}")
