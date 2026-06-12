from finetuning.lora import LoRALinear


def inject_lora(
    model,
    rank=8,
    alpha=16,
):

    for block in model.layers:

        block.attn.q_proj = LoRALinear(
            block.attn.q_proj,
            rank,
            alpha,
        )

        block.attn.k_proj = LoRALinear(
            block.attn.k_proj,
            rank,
            alpha,
        )

        block.attn.v_proj = LoRALinear(
            block.attn.v_proj,
            rank,
            alpha,
        )

        block.attn.out_proj = LoRALinear(
            block.attn.out_proj,
            rank,
            alpha,
        )

    return model



def mark_only_lora_trainable(model):

    for p in model.parameters():
        p.requires_grad = False

    for module in model.modules():

        if hasattr(module, "A"):
            module.A.requires_grad = True

        if hasattr(module, "B"):
            module.B.requires_grad = True

def inject_lora(
    model,
    rank=8,
    alpha=16,
):

    for block in model.layers:

        block.attn.q_proj = LoRALinear(
            block.attn.q_proj,
            rank,
            alpha,
        )

        block.attn.k_proj = LoRALinear(
            block.attn.k_proj,
            rank,
            alpha,
        )

        block.attn.v_proj = LoRALinear(
            block.attn.v_proj,
            rank,
            alpha,
        )

        block.attn.out_proj = LoRALinear(
            block.attn.out_proj,
            rank,
            alpha,
        )

    return model


def print_trainable_params(model):

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Trainable: {trainable:,}"
    )

    print(
        f"Total: {total:,}"
    )

    print(
        f"% Trainable: "
        f"{100 * trainable / total:.4f}%"
    )