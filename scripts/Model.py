import json
import re
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("punkt")

# Initialize BERT model and tokenizer
# Note: Use the `cache_dir` parameter if you encounter issues with the model downloading
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
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Ensure 768-dim

def calculate_bleu(reference, candidate):
    """Calculate BLEU score between reference and candidate texts."""
    ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference]
    cand_tokens = nltk.word_tokenize(candidate.lower())
    smoothie = SmoothingFunction().method4
    return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smoothie)

def extract_agent_data(file_path):
    """Extract agent outputs and calculate embeddings and BLEU scores."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    agent_keys = ['cot', 'anchoring', 'associate', 'logician', 'cognition']
    true_answer = data['true_answer']
    agent_data = {}

    for key in agent_keys:
        if key in data:
            output = preprocess_text(data[key][f'{key}_output'])
            embedding = get_embedding(output)
            bleu_score = calculate_bleu(true_answer, output)
            agent_data[key] = {'embedding': embedding, 'bleu_score': bleu_score}

    target_embedding = get_embedding(' '.join(true_answer))
    return agent_data, target_embedding

class AgentDataset(Dataset):
    """Custom dataset to load agent data and target embeddings."""
    def __init__(self, json_files):
        self.data = [extract_agent_data(file) for file in json_files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        agent_data, target = self.data[idx]
        embeddings = np.array([agent_data[a]['embedding'] for a in agent_data])
        bleu_scores = np.array([[agent_data[a]['bleu_score']] for a in agent_data])

        return (
            torch.tensor(embeddings, dtype=torch.float32),
            torch.tensor(bleu_scores, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            agent_data
        )

class AgentWeightingModel(nn.Module):
    """Neural network model to learn agent weights."""
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
    """Ensure embeddings are of the same shape and find the closest agent."""
    point_in_space = point_in_space.reshape(1, -1)  # Ensure 2D shape

    similarities = {
        agent: cosine_similarity(embedding.reshape(1, -1)[:, :768], point_in_space)[0][0]
        for agent, embedding in agent_embeddings.items()
    }

    closest_agent = max(similarities, key=similarities.get)
    return closest_agent, similarities

# Load all JSON files from the specified directory
json_dir = "logs/triviaqa/top10"
all_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]

# Split the dataset into 80% train and 20% test
train_size = int(0.8 * len(all_files))
test_size = len(all_files) - train_size
train_files, test_files = random_split(all_files, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(AgentDataset(train_files), batch_size=2, shuffle=True)
test_loader = DataLoader(AgentDataset(test_files), batch_size=1, shuffle=False)

# Initialize model, loss function, and optimizer
model = AgentWeightingModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for embeddings, scores, target, agent_data in train_loader:
        # Forward pass to get output and weights
        output, weights = model(embeddings, scores)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Find the closest agent during training
        agent_embeddings = {agent: data['embedding'] for agent, data in agent_data.items()}
        closest_agent, similarities = find_closest_agent(agent_embeddings, output[0].detach().numpy())

        # Print training results
        agents = list(agent_data.keys())
        print(f"Training - Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")
        print(f"Closest Agent: {closest_agent}, Similarities: {similarities}")
        for i, agent in enumerate(agents):
            print(f"Weight for {agent}: {weights[0, i].item():.4f}")

# Testing loop
model.eval()
with torch.no_grad():
    for embeddings, scores, target, agent_data in test_loader:
        # Forward pass to get output and weights
        output, weights = model(embeddings, scores)
        test_loss = criterion(output, target)

        # Find the closest agent during testing
        agent_embeddings = {agent: data['embedding'] for agent, data in agent_data.items()}
        closest_agent, similarities = find_closest_agent(agent_embeddings, output[0].detach().numpy())

        # Print test results
        agents = list(agent_data.keys())
        print(f"Test - Closest Agent: {closest_agent}, Similarities: {similarities}")
        print(f'Test Loss: {test_loss.item():.4f}')
        for i, agent in enumerate(agents):
            print(f"Weight for {agent}: {weights[0, i].item():.4f}")
