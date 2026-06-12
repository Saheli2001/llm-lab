import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT))

import torch
from config import Config
from utils import download_dataset, load_text, build_tokenizer
from dataset import ShakespeareDataset
from model import ModernMiniGPT
from train import train
from finetuning.inject_lora import (
    inject_lora,
    mark_only_lora_trainable, print_trainable_params

)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Config.device = device

    # Download dataset
    download_dataset(
        Config.dataset_url,
        Config.dataset_path,
    )

    # Load dataset
    text = load_text(Config.dataset_path)

    # Tokenizer
    stoi, itos, encode, decode = build_tokenizer(text)
    dataset = ShakespeareDataset(
        text,
        encode,
        Config.block_size,
    )

    # Model
    model = ModernMiniGPT(
        vocab_size=len(stoi),
        block_size=Config.block_size,
        dim=Config.dim,
        n_layers=Config.n_layers,
        n_heads=Config.n_heads,
        n_kv_heads=Config.n_kv_heads,
        hidden_dim=Config.hidden_dim,
        dropout=Config.dropout,
    ).to(device)

    if Config.is_lora==True:
        model = inject_lora(
            model,
            rank=8,
            alpha=16,
        )

        mark_only_lora_trainable(model)
        print_trainable_params(model)

    train(model, dataset, decode, Config.is_lora)
        


if __name__ == "__main__":
    main()