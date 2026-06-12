import os
import requests
import torch

def download_dataset(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        print("Downloading dataset...")
        text = requests.get(url).text
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Dataset downloaded.")

def build_tokenizer(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(tokens):
        return "".join([itos[i] for i in tokens])
    
    return stoi, itos, encode, decode

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()