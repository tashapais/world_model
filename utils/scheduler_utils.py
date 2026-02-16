import math
import torch.optim as optim

def cosine_with_warmup(step, *, warmup_steps, total_steps, min_lr=0.0):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    # calculate step within warmup
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

    # calculate where on cosine curve we are
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (1 - min_lr) * cosine

def create_cosine_scheduler(optimizer, total_steps, warmup_fraction=0.05, min_lr_fraction=0.01):
    warmup_steps = int(total_steps * warmup_fraction)
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(
            s,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr_fraction
        )
    ) 