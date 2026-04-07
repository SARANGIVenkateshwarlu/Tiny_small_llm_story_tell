<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# here is the colab small llm code: keep everything same flow from dataset to training and add optuna fuction for fine tune, optimization, and for various parameters combinations and decrease model loss and increase accuracy, parameters, n-layers, n-heads, dropout, bias = true or false, learning rate, decay etc.,. iteration = upto 100000, each time update best one. and every time best params save file and weights of the model. Model will be run on lightining .AI rented GPU

Yes — you can keep the same notebook flow (dataset → tokenization → batching → model → loss → training → inference) and add an Optuna-based hyperparameter search around the training section, with automatic best-trial tracking, checkpoint saving, and best-weight export for Lightning AI GPUs.[^1]

## What to change

Your notebook already has the full end-to-end pipeline: TinyStories loading, GPT-2 tokenizer-based `.bin` creation, `get_batch`, a configurable GPT architecture with `n_layer`, `n_head`, `dropout`, and `bias`, plus a training loop that already saves the best validation weights to `best_model_params.pt`.[^1]
The cleanest approach is to leave Steps 1–5 almost unchanged, refactor Steps 6–8 into reusable functions, and then let Optuna call those functions with different hyperparameter combinations such as `n_layer`, `n_head`, `dropout`, `bias`, `learning_rate`, `weight_decay`, warmup, batch size, and gradient clipping.[^1]

## Important fixes

Your current scheduler section has `min_lr = 5e-4` while `learning_rate = 1e-4`, which makes the minimum learning rate larger than the starting rate and works against cosine decay.[^1]
The current training loop also triggers the PyTorch warning that `lr_scheduler.step()` is being called before the optimizer step order is fully correct, so that should be cleaned up inside the refactor to avoid unstable learning-rate behavior during trials.[^1]

## Lightning AI setup

Because you want to run on rented Lightning AI GPU, save everything to a persistent directory such as `/teamspace/studios/this_studio/checkpoints/` or another mounted workspace path, not only the current runtime folder, otherwise trial artifacts may disappear when the session restarts.
For long searches, do not run all trials to 100000 full iterations from the start; use Optuna pruning with intermediate validation loss reports, then let only the best configuration run the full `max_iters=100000`, which is much more GPU-efficient while still updating the best parameters every time a stronger trial appears.[^1]

## Recommended structure

Use this flow:

1. Keep dataset download and tokenization exactly once.[^1]
2. Keep `GPTConfig`, `GPT`, `get_batch`, and `estimate_loss`, but make them parameter-driven.[^1]
3. Add an `objective(trial)` function that:

- samples hyperparameters,
- builds a model,
- trains for a shorter search budget,
- reports validation loss to Optuna,
- prunes weak trials early,
- saves `best_trial.json`, `best_model.pt`, and optionally optimizer/scheduler/scaler state whenever a new global best is found.

4. After study completion, rebuild the model using the best params and run a final long training phase up to 100000 iterations.[^1]

## Drop-in code

Add a new section after your model definition and before the final inference section. This keeps your notebook flow intact while replacing the single fixed training config with Optuna search.[^1]

```python
!pip install -q optuna

import os
import json
import math
import time
import copy
import random
import optuna
import numpy as np
import torch
from contextlib import nullcontext
from dataclasses import asdict

SAVE_DIR = "/teamspace/studios/this_studio/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

vocab_size = 50257
block_size = 128

global_best_val = float("inf")
global_best_path = os.path.join(SAVE_DIR, "best_model_params.pt")
global_best_meta_path = os.path.join(SAVE_DIR, "best_trial.json")
study_db = f"sqlite:///{SAVE_DIR}/optuna_study.db"

def make_model_from_trial_params(params):
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=params["block_size"],
        n_layer=params["n_layer"],
        n_head=params["n_head"],
        n_embd=params["n_embd"],
        dropout=params["dropout"],
        bias=params["bias"],
    )
    model = GPT(config).to(device)
    return model, config

def make_optimizer_and_scheduler(model, params):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        betas=(0.9, params["beta2"]),
        weight_decay=params["weight_decay"],
        eps=1e-8
    )

    warmup_steps = params["warmup_steps"]
    max_iters = params["max_iters"]
    min_lr = params["min_lr"]

    def get_lr(it):
        if it < warmup_steps:
            return params["learning_rate"] * (it + 1) / warmup_steps
        if it > max_iters:
            return min_lr
        decay_ratio = (it - warmup_steps) / max(1, (max_iters - warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (params["learning_rate"] - min_lr)

    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda" and dtype == "float16"))
    return optimizer, scaler, get_lr

def estimate_loss_for_model(model, eval_iters, batch_size_local, block_size_local):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch_custom(split, batch_size_local, block_size_local)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    model.train()
    return out

def get_batch_custom(split, batch_size_local, block_size_local):
    if split == "train":
        data = np.memmap("train.bin", dtype=np.uint16, mode="r")
    else:
        data = np.memmap("validation.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size_local, (batch_size_local,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size_local]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size_local]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def save_global_best(model, optimizer, scaler, config, params, iter_num, val_loss, trial_number):
    payload = {
        "trial_number": trial_number,
        "iter_num": iter_num,
        "val_loss": val_loss,
        "params": params,
        "config": asdict(config),
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": asdict(config),
        "params": params,
        "iter_num": iter_num,
        "val_loss": val_loss,
        "trial_number": trial_number,
    }, global_best_path)

    with open(global_best_meta_path, "w") as f:
        json.dump(payload, f, indent=2)

def objective(trial):
    global global_best_val

    n_head = trial.suggest_categorical("n_head", [4, 6, 8])
    head_dim = trial.suggest_categorical("head_dim", [48, 64, 80])
    n_embd = n_head * head_dim

    params = {
        "block_size": 128,
        "n_layer": trial.suggest_int("n_layer", 4, 10),
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": trial.suggest_float("dropout", 0.05, 0.25),
        "bias": trial.suggest_categorical("bias", [True, False]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 8e-4, log=True),
        "min_lr": trial.suggest_float("min_lr", 1e-5, 8e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 2e-1, log=True),
        "beta2": trial.suggest_float("beta2", 0.95, 0.999),
        "warmup_steps": trial.suggest_int("warmup_steps", 200, 3000),
        "batch_size": trial.suggest_categorical("batch_size", [16, 24, 32]),
        "grad_accum": trial.suggest_categorical("grad_accum", [8, 16, 32]),
        "grad_clip": trial.suggest_float("grad_clip", 0.3, 1.0),
        "max_iters": 12000,
        "eval_interval": 500,
        "eval_iters": 100,
    }

    model, config = make_model_from_trial_params(params)
    optimizer, scaler, get_lr = make_optimizer_and_scheduler(model, params)

    best_trial_val = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for iter_num in range(params["max_iters"]):
        lr = get_lr(iter_num)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        X, Y = get_batch_custom("train", params["batch_size"], params["block_size"])

        with ctx:
            _, loss = model(X, Y)
            loss = loss / params["grad_accum"]

        scaler.scale(loss).backward()

        if (iter_num + 1) % params["grad_accum"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if iter_num % params["eval_interval"] == 0 and iter_num > 0:
            losses = estimate_loss_for_model(
                model,
                eval_iters=params["eval_iters"],
                batch_size_local=params["batch_size"],
                block_size_local=params["block_size"],
            )
            val_loss = losses["val"]
            trial.report(val_loss, step=iter_num)

            if val_loss < best_trial_val:
                best_trial_val = val_loss

            if val_loss < global_best_val:
                global_best_val = val_loss
                save_global_best(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    params=params,
                    iter_num=iter_num,
                    val_loss=val_loss,
                    trial_number=trial.number,
                )

            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_trial_val
```

Then run the study:

```python
sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)

study = optuna.create_study(
    study_name="slm_tinystories_optimization",
    storage=study_db,
    load_if_exists=True,
    direction="minimize",
    sampler=sampler,
    pruner=pruner,
)

study.optimize(objective, n_trials=30, gc_after_trial=True)

print("Best trial:", study.best_trial.number)
print("Best val loss:", study.best_value)
print("Best params:", study.best_params)
```

Then do a final long run with the best params up to 100000 iterations:

```python
best = study.best_trial.params

n_embd = best["n_head"] * best["head_dim"]
final_params = {
    "block_size": 128,
    "n_layer": best["n_layer"],
    "n_head": best["n_head"],
    "n_embd": n_embd,
    "dropout": best["dropout"],
    "bias": best["bias"],
    "learning_rate": best["learning_rate"],
    "min_lr": best["min_lr"],
    "weight_decay": best["weight_decay"],
    "beta2": best["beta2"],
    "warmup_steps": best["warmup_steps"],
    "batch_size": best["batch_size"],
    "grad_accum": best["grad_accum"],
    "grad_clip": best["grad_clip"],
    "max_iters": 100000,
    "eval_interval": 1000,
    "eval_iters": 200,
}

model, config = make_model_from_trial_params(final_params)
optimizer, scaler, get_lr = make_optimizer_and_scheduler(model, final_params)

best_val_loss = float("inf")
optimizer.zero_grad(set_to_none=True)

for iter_num in range(final_params["max_iters"]):
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    X, Y = get_batch_custom("train", final_params["batch_size"], final_params["block_size"])

    with ctx:
        _, loss = model(X, Y)
        loss = loss / final_params["grad_accum"]

    scaler.scale(loss).backward()

    if (iter_num + 1) % final_params["grad_accum"] == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), final_params["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if iter_num % final_params["eval_interval"] == 0 and iter_num > 0:
        losses = estimate_loss_for_model(
            model,
            eval_iters=final_params["eval_iters"],
            batch_size_local=final_params["batch_size"],
            block_size_local=final_params["block_size"],
        )
        print(f"iter {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {lr:.6f}")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "params": final_params,
                "iter_num": iter_num,
                "val_loss": best_val_loss,
            }, os.path.join(SAVE_DIR, "final_best_model.pt"))

            with open(os.path.join(SAVE_DIR, "final_best_model.json"), "w") as f:
                json.dump({
                    "config": asdict(config),
                    "params": final_params,
                    "iter_num": iter_num,
                    "val_loss": best_val_loss,
                }, f, indent=2)
```

For inference, load from the checkpoint dict instead of raw state dict:

```python
ckpt = torch.load(os.path.join(SAVE_DIR, "final_best_model.pt"), map_location=device)
config = GPTConfig(**ckpt["config"])
model = GPT(config).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```


## Hyperparameter notes

Because your model class requires `n_embd % n_head == 0`, sample `n_head` and `head_dim`, then compute `n_embd = n_head * head_dim`; that avoids invalid combinations during Optuna trials.[^1]
Also, searching `n_layer`, `n_head`, dropout, bias, learning rate, weight decay, warmup, and gradient accumulation is aligned with the architecture and optimizer knobs already present in the notebook, so you are extending the existing code rather than rewriting the pipeline.[^1]

## Best practical settings

- Use short search trials first, for example 8k–12k iterations, then one final 100k run on the best trial.[^1]
- Save checkpoints to persistent Lightning storage every time global best validation loss improves.[^1]
- Add `study_db` SQLite persistence so the optimization can resume after disconnects.[^1]
- Track both loss and a token-level next-token accuracy metric if you want “accuracy” explicitly, since your current notebook optimizes only cross-entropy loss.[^1]

Would you like me to turn this into a fully edited `.ipynb` version of your notebook with the Optuna cells inserted in the right places?

<div align="center">⁂</div>

[^1]: Copy_of_Vizuara_AI_Labs_Small_Language_Model_Scratch_Final.ipynb

