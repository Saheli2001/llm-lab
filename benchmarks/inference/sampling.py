import torch


def greedy_sampling(logits):

    return torch.argmax(
        logits,
        dim=-1,
    )


def top_k_sampling(logits, k=50):

    values, indices = torch.topk(
        logits,
        k,
    )

    probs = torch.softmax(
        values,
        dim=-1,
    )

    sampled = torch.multinomial(
        probs,
        num_samples=1,
    )

    return indices.gather(
        -1,
        sampled,
    )


def top_p_sampling(logits, p=0.9):

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )

    cumulative_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1),
        dim=-1,
    )

    mask = cumulative_probs > p

    sorted_logits[mask] = -float("inf")

    probs = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    sampled = torch.multinomial(
        probs,
        num_samples=1,
    )

    return sorted_indices.gather(
        -1,
        sampled,
    )