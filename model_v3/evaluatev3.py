import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Re-create the same preprocessing steps as training
OUTPUT_DIR = "shakespeare_works"
combined_text_file = "data_shakespeare.txt"

with open(combined_text_file, "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# Use the same parameters as training
vocab_size = len(chars)
embedding_dim = 256
rnn_units = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same model
class ShakespeareModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(ShakespeareModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.lstm2 = nn.LSTM(rnn_units, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x

model = ShakespeareModel(vocab_size, embedding_dim, rnn_units).to(device)

# Load the model weights
model.load_state_dict(torch.load("shakespeare_generator.pth", map_location=device))
model.eval()

chars = sorted(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

text_as_int = np.array([char_to_idx[c] for c in text])

seq_length = 100  # Example, ensure this matches your training seq_length
split_ratio = 0.9
split_index = int(len(text_as_int) * split_ratio)

train_data = text_as_int[:split_index]
val_data = text_as_int[split_index:]

if len(val_data) <= seq_length:
    raise ValueError("Validation data is too small. Increase your dataset or reduce seq_length.")

val_inputs = []
val_targets = []
examples_per_epoch_val = len(val_data) - seq_length
for i in range(examples_per_epoch_val):
    val_inputs.append(val_data[i:i+seq_length])
    val_targets.append(val_data[i+1:i+1+seq_length])

val_inputs = np.array(val_inputs)
val_targets = np.array(val_targets)

# Step 5: Create Validation DataLoader
from torch.utils.data import Dataset, DataLoader
import torch

class ShakespeareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

val_dataset = ShakespeareDataset(val_inputs, val_targets)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
criterion = nn.CrossEntropyLoss()

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    count = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            outputs = model(batch_inputs)  # [batch, seq_length, vocab_size]

            outputs_flat = outputs.view(-1, vocab_size)
            targets_flat = batch_targets.view(-1)
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            count += 1

            # Compute character-level accuracy
            predictions = torch.argmax(outputs, dim=-1)
            correct = (predictions == batch_targets).sum().item()
            total_correct += correct
            total_tokens += targets_flat.numel()

    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)
    accuracy = (total_correct / total_tokens) * 100
    return avg_loss, perplexity, accuracy

val_loss, val_perplexity, val_accuracy = evaluate_model(model, val_loader)
print(f"Validation Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}, Accuracy: {val_accuracy:.2f}%")

def generate_text(model, start_string, num_generate=500, temperature=1.0):
    model.eval()
    input_eval = torch.tensor([char_to_idx[char] for char in start_string], dtype=torch.long).unsqueeze(0).to(device)
    text_generated = []

    with torch.no_grad():
        for _ in range(num_generate):
            outputs = model(input_eval)  # [1, seq_length, vocab_size]
            predictions = outputs[:, -1, :] / temperature
            probs = torch.softmax(predictions, dim=-1)
            predicted_id = torch.multinomial(probs, num_samples=1).item()

            text_generated.append(idx_to_char[predicted_id])
            input_eval = torch.cat([input_eval[:, 1:], torch.tensor([[predicted_id]], device=device)], dim=1)

    return start_string + ''.join(text_generated)

seed_text = "Romeo and Juliet"
sample = generate_text(model, seed_text, num_generate=500, temperature=1.0)
print("Sample generated text:\n", sample)
