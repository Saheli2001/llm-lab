class Config:
    dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    dataset_path = "data/input.txt"

    batch_size = 32
    block_size = 256
    max_iters = 1000
    eval_interval = 100
    learning_rate = 3e-4
    weight_decay = 0.01
    grad_clip = 1.0

    dim = 512
    n_layers = 8
    n_heads = 8
    n_kv_heads = 2
    hidden_dim = 1536
    dropout = 0.0

    device = "gpu:1"
    is_lora = True