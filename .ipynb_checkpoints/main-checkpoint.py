import torch
from config import Config
from utils import download_dataset, load_text, build_tokenizer
from dataset import ShakespeareDataset
from model import ModernMiniGPT
from train import train

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

    train(model, dataset, decode)


if __name__ == "__main__":
    main()