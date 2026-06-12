import torch

class ShakespeareDataset:
    def __init__(self, text, encode, block_size):
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9*len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        self.block_size = block_size

    def get_batch(self, split, batch_size, device):
        source = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(source)-self.block_size, (batch_size,))
        x = torch.stack([source[i:i+self.block_size] for i in ix])
        y = torch.stack([source[i+1:i+self.block_size+1] for i in ix])
        return x.to(device), y.to(device)
    