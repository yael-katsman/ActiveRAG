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

epochs = 10
learning_rate = 0.001
datasets = ["triviaqa", "popqa", "nq", "webq"]
topk_values = [5, 10]

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
            nn.Sigmoid()  # Following small model's Sigmoid approach
        )

    def forward(self, agent_embeddings, bleu_scores):
        inputs = torch.cat((agent_embeddings, bleu_scores), dim=-1)
        logits = self.fc(inputs).squeeze(-1)
        weighted_embeddings = (logits.unsqueeze(-1) * agent_embeddings).sum(dim=1)
        return weighted_embeddings, logits

def find_closest_agent(agent_embeddings, point_in_space):
    point_in_space = point_in_space.reshape(1, -1)
    similarities = {
        agent: cosine_similarity(np.array(embedding).reshape(1, -1), point_in_space)[0][0]
        for agent, embedding in agent_embeddings.items()
    }
    return max(similarities, key=similarities.get)

# Training and Testing Loop
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
                logits = model(embeddings, scores)[1]  # Get logits
                target_indices = torch.tensor([list(agent_data.keys()).index(find_closest_agent(agent_data, embeddings.mean(dim=1).cpu().numpy()))])
                loss = criterion(logits, target_indices)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Dataset: {dataset}, TopK: {topk}, Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        # Testing loop
        model.eval()
        all_test_results = []
        with torch.no_grad():
            for embeddings, scores, agent_data, filename_tuple in test_loader:
                logits = model(embeddings, scores)[1]
                chosen_agent_idx = logits.argmax(dim=-1).item()
                chosen_agent = list(agent_data.keys())[chosen_agent_idx]
                chosen_agent_correctness = agent_data[chosen_agent]['correctness']

                result = {
                    'filename': os.path.basename(filename_tuple[0]),
                    'chosen_agent': chosen_agent,
                    'output': agent_data[chosen_agent]['output'],
                    'bleu_score': float(agent_data[chosen_agent]['bleu_score']),
                    'correctness': str(chosen_agent_correctness),
                    'weights': logits.cpu().numpy().tolist()
                }
                all_test_results.append(result)

        output_dir = f"Model_Answers/{dataset}/top{topk}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'results_big_model.json')

        with open(output_file, 'w') as f:
            json.dump(all_test_results, f, indent=4)

        # Calculate metrics for the summary
        accuracy = calc_accuracy(output_file)
        average_bleu_score = calc_average_bleu_score(output_file)
        agent_histogram = get_agent_histogram(output_file)

        # Save summary in summary_big_model.json
        summary = {
            'accuracy': f"{accuracy:.2f}%",
            'average_bleu_score': average_bleu_score,
            'agent_histogram': agent_histogram,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'loss_function': "CrossEntropyLoss"
        }

        summary_file = os.path.join(output_dir, 'summary_big_model.json')

        # Append or initialize summary
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

        print(f"Summary metrics for {dataset} (top {topk}) saved in: {summary_file}")
