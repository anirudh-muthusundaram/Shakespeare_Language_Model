import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

OUTPUT_DIR = "shakespeare_works"
combined_text_file = "data_shakespeare.txt"

# Combine all Shakespeare texts
combined_text = ""
for filename in os.listdir(OUTPUT_DIR):
    with open(os.path.join(OUTPUT_DIR, filename), "r", encoding="utf-8") as file:
        combined_text += file.read().strip() + "\n\n"

with open(combined_text_file, "w", encoding="utf-8") as file:
    file.write(combined_text)
print(f"Combined texts saved to {combined_text_file}")

# Load combined text
with open(combined_text_file, "r", encoding="utf-8") as file:
    text = file.read()

# Create character mappings
chars = sorted(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# Convert text to int representation
text_as_int = np.array([char_to_idx[char] for char in text])

# Sequence preparation
seq_length = 100
examples_per_epoch = len(text) - seq_length

# Create inputs and targets so that targets are inputs shifted by one character
inputs = []
targets = []
for i in range(examples_per_epoch):
    inputs.append(text_as_int[i:i+seq_length])
    targets.append(text_as_int[i+1:i+1+seq_length])

inputs = np.array(inputs)
targets = np.array(targets)

class ShakespeareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = ShakespeareDataset(inputs, targets)

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

class ShakespeareModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(ShakespeareModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.lstm2 = nn.LSTM(rnn_units, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)           # [batch, seq_length, embedding_dim]
        x, _ = self.lstm1(x)            # [batch, seq_length, rnn_units]
        x, _ = self.lstm2(x)            # [batch, seq_length, rnn_units]
        x = self.fc(x)                  # [batch, seq_length, vocab_size]
        return x

vocab_size = len(chars)
embedding_dim = 256
rnn_units = 512

model = ShakespeareModel(vocab_size, embedding_dim, rnn_units).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
loss_history = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_inputs, batch_targets in dataloader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)         # [batch, seq_length, vocab_size]
        
        # Flatten for loss calculation
        outputs = outputs.view(-1, vocab_size)       # [batch*seq_length, vocab_size]
        batch_targets = batch_targets.view(-1)       # [batch*seq_length]

        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Function to generate text
def generate_text(model, start_string, num_generate=500, temperature=1.0):
    model.eval()
    input_eval = torch.tensor([char_to_idx[char] for char in start_string], dtype=torch.long).unsqueeze(0).to(device)
    text_generated = []

    with torch.no_grad():
        for _ in range(num_generate):
            outputs = model(input_eval)                 # [1, seq_length, vocab_size]
            predictions = outputs[:, -1, :] / temperature
            probs = torch.softmax(predictions, dim=-1)
            predicted_id = torch.multinomial(probs, num_samples=1).item()

            text_generated.append(idx_to_char[predicted_id])
            input_eval = torch.cat([input_eval[:, 1:], torch.tensor([[predicted_id]], device=device)], dim=1)

    return start_string + ''.join(text_generated)

# Test generation
seed_text = "To be, or not to be, that is the question:"
generated_text = generate_text(model, seed_text, num_generate=1000)
print(generated_text)

import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# Save the model for future use
torch.save(model.state_dict(), "shakespeare_generator.pth")
print("Model weights saved as shakespeare_generator.pth")

model = ShakespeareModel(vocab_size, embedding_dim, rnn_units)
model.load_state_dict(torch.load("shakespeare_generator.pth", map_location=device))
model.eval()
