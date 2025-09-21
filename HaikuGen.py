import random
import string

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Data preprocessing utilities
# -----------------------------
def clean_lines(filepath="lines.txt"):
    """
    Load, clean, and pad s from a txt file.

    - Converts haikus to lowercase.
    - Appends an end-of-string marker.
    - Pads all lines to the maximum length with the EOS token.
    """
    lines = pd.read_csv(filepath)  # from https://www.kaggle.com/datasets/bfbarry/haiku-dataset
    lines.columns = ['poem']
    # print(lines.head())
    poem_list = [poem.lower() for poem in lines["poem"] if check_special_char(poem)]
    max_length = max(len(poem) for poem in poem_list)
    poem_list = [poem.ljust(max_length, '$') for poem in poem_list]
    return poem_list, max_length

def check_special_char(poem):
    special_characters = ["$", "/", " "]
    allowed_chars = list(string.ascii_lowercase)+special_characters
    for c in poem:
        if not any(sp in c for sp in allowed_chars):
            return False
    return True

def get_dictionaries():
    """
    Return dictionaries for character-index conversions.
    """
    characters = list(string.ascii_lowercase) + ["$", "/", " "]
    alpha_to_idx = {char: idx for idx, char in enumerate(characters)}
    idx_to_alpha = {idx: char for idx, char in enumerate(characters)}
    return alpha_to_idx, idx_to_alpha


def encode_haikus(haikus, alpha_to_idx):
    """
    Convert a list of haikus into one-hot encoded tensors.
    Shape: (num_haikus, max_length, vocab_size)
    """
    idx_matrix = torch.tensor([[alpha_to_idx[char] for char in haiku] for haiku in haikus])
    return nn.functional.one_hot(idx_matrix, num_classes=len(alpha_to_idx)).float()


# -----------------------------
# Dataset
# -----------------------------
class HaikuDataset(Dataset):
    """
    Custom Dataset for haikus completion.
    Each item is a one-hot encoded haiku.
    """

    def __init__(self, filepath="lines.txt"):
        lines, self.max_length = clean_lines(filepath)
        self.alpha_to_idx, self.idx_to_alpha = get_dictionaries()
        self.haikus_one_hot = encode_haikus(lines, self.alpha_to_idx)

    def __len__(self):
        return len(self.haikus_one_hot)

    def __getitem__(self, idx):
        return self.haikus_one_hot[idx]


# -----------------------------
# Model
# -----------------------------
class HaikuCompleter(nn.Module):
    """
    GRU-based model that predicts the next character in a haiku.
    """

    def __init__(self, hidden_dim=128, vocab_size=29, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # Creates a fully connected GRU into linear layer
        self.gru = nn.GRU(vocab_size,
                          hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=0.05
                          )
        self.fc = nn.Linear(hidden_dim*2, vocab_size)


    def forward(self, x):
        # hn: hidden state from last time step
        _, hn = self.gru(x)
        return self.fc(torch.cat((hn[-1], hn[-2]), dim=1))  # (batch_size, vocab_size)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)

# -----------------------------
# Haiku generation
# -----------------------------
def generate_haiku(prefix, model, idx_to_alpha, alpha_to_idx, max_length, temperature=1.0):
    """
    Generate a haiku starting from a prefix using temperature sampling.

    Args:
        prefix (str): Starting string.
        model (nn.Module): Trained model.
        idx_to_alpha (dict): Index-to-character mapping.
        alpha_to_idx (dict): Character-to-index mapping.
        temperature (float): Controls randomness:
                             - temperature < 1 → sharper, more greedy
                             - temperature > 1 → more random
    """
    model.eval()
    haiku = prefix.lower()

    for _ in range(max_length - len(prefix)):
        encoded = encode_haikus([haiku], alpha_to_idx)
        with torch.no_grad():
            logits = model(encoded)

            # Apply temperature
            logits = logits / temperature

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx_to_alpha[next_idx]

        if next_char == '$':
            break
        haiku += next_char

    return haiku

def generate_haikus(model, idx_to_alpha, alpha_to_idx, max_length, temperature=1.0):
    """
    Generate haikus starting from each letter of the alphabet.
    """
    return [generate_haiku(ch, model, idx_to_alpha, alpha_to_idx,max_length=max_length, temperature=temperature) for ch in string.ascii_lowercase]

def generate_haiku_from_word(word, model, idx_to_alpha, alpha_to_idx, max_length, temperature=1.0):
    """
    Generate haikus starting from each letter of the alphabet.
    """
    return [generate_haiku(word, model, idx_to_alpha, alpha_to_idx,max_length=max_length, temperature=temperature)]

# -----------------------------
# Training loop
# -----------------------------
def train_model(
        epochs=150,
        batch_size=64,
        min_length=1,
        lr=5e-3,
        hidden_dim=128,
        num_layers = 2,
        filepath="lines.txt"
):
    """
    Train the RNN model on haikus and print sample generations.
    """
    # Initialize dataset and model
    dataset = HaikuDataset(filepath)
    alpha_to_idx, idx_to_alpha = dataset.alpha_to_idx, dataset.idx_to_alpha
    max_length = dataset.max_length
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = HaikuCompleter(hidden_dim=hidden_dim, vocab_size=len(alpha_to_idx), num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()

        for batch in dataloader:
            # Randomly choose a cutoff length
            length = random.randint(min_length, batch.shape[1] - 1)
            inputs = batch[:, :length]  # prefix
            targets = batch[:, length]  # true next character

            optimizer.zero_grad()
            outputs = model(inputs)  # logits
            loss = loss_fn(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        # Show sample generations
        temperature = 0.25
        generated_haikus = generate_haikus(model, idx_to_alpha, alpha_to_idx,max_length=max_length, temperature=temperature)
        print(f"------ Generated Haikus (Temperature = {temperature}) ------")
        for i in range(26):
            print(generated_haikus[i])
        # Print out loss
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss: .4f}")
        print()

    return model, alpha_to_idx, idx_to_alpha


# -----------------------------
# Run training
# -----------------------------
if __name__ == "__main__":
    # model, _, _ = train_model(hidden_dim=128, num_layers = 2)
    # torch.save(model.state_dict(), 'haiku_model_dropout.pt')

    haiku_model = HaikuCompleter(hidden_dim=128, num_layers=2)
    haiku_model.load_state_dict(torch.load('haiku_model_dropout.pt', map_location=(torch.device('cpu'))))

    alpha_to_idx, idx_to_alpha = get_dictionaries()
    starting_word = input("Enter starting word for Haiku: ")
    while starting_word != 'end':
        for i in range(1, 5):
            temperature = float(i)/10
            generated_haiku = generate_haiku_from_word(starting_word, haiku_model, idx_to_alpha, alpha_to_idx, max_length=85,
                                           temperature=temperature)
            print(f"Generated Haiku (Temperature = {temperature})")
            print(generated_haiku)
            print()

        starting_word = input("Enter starting word for Haiku: ")
