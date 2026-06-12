import torch


class KVCache:

    def __init__(
        self,
        batch_size,
        max_seq_len,
        n_kv_heads,
        head_dim,
        device,
        dtype=torch.float16,
    ):

        self.k_cache = torch.zeros(
            batch_size,
            n_kv_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

        self.v_cache = torch.zeros(
            batch_size,
            n_kv_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

    def update(self, k, v, position):

        self.k_cache[:, :, position:position+1] = k
        self.v_cache[:, :, position:position+1] = v

    def get_cache(self, upto_idx):

        return (
            self.k_cache[:, :, :upto_idx],
            self.v_cache[:, :, :upto_idx],
        )