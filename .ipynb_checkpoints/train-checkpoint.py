import torch
from config import Config
from utils import save_checkpoint


@torch.no_grad()
def estimate_loss(model, dataset):
    """
    Evaluate train and validation loss.
    """
    out = {}
    model.eval()

    for split in ["train", "val"]:

        losses = torch.zeros(20)

        for k in range(20):

            X, Y = dataset.get_batch(
                split=split,
                batch_size=Config.batch_size,
                device=Config.device,
            )

            _, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


def train(model, dataset, decode):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(Config.device == "cuda")
    )

    for step in range(Config.max_iters):

        # =========================
        # Evaluation
        # =========================
        if step % Config.eval_interval == 0:

            losses = estimate_loss(model, dataset)

            print(
                f"[Step {step}] "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f}"
            )

        # =========================
        # Get batch
        # =========================
        xb, yb = dataset.get_batch(
            split="train",
            batch_size=Config.batch_size,
            device=Config.device,
        )

        # =========================
        # Forward pass
        # =========================
        with torch.autocast(
            device_type="cuda" if Config.device == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=(Config.device == "cuda")
        ):

            _, loss = model(xb, yb)

        # =========================
        # Backpropagation
        # =========================
        optimizer.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            Config.grad_clip,
        )

        scaler.step(optimizer)

        scaler.update()

    # =========================
    # Save model
    # =========================
    save_checkpoint(
        model,
        "modern_mini_gpt.pth"
    )

    print("\nModel checkpoint saved.")

    # =========================
    # Generate sample text
    # =========================
    context = torch.zeros(
        (1, 1),
        dtype=torch.long,
        device=Config.device,
    )

    generated = model.generate(
        context,
        max_new_tokens=500,
        temperature=0.8,
        top_k=50,
    )[0].tolist()

    print("\n================ GENERATED TEXT ================\n")

    print(decode(generated))

    print("\n================================================\n")