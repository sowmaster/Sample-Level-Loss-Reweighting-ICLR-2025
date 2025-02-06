# Loss-based sample-level reweighting for LLM pre-training
This repository contains the code for the ICLR 2025 paper 
**Dynamic Loss-Based Sample Reweighting for Improved Large Language Model Pretraining**. 

[![Paper](https://img.shields.io/badge/OpenReview-ICLR2025-b31b1b.svg)](https://openreview.net/forum?id=gU4ZgQNsOC)


## Setup
Install the requirements inside `requirements.txt`

## Experiment configurations
You may find all configuration files in the `config/` folder of this repository.

**NOTE REGARDING 1.4B AND 7B EXPERIMENTS:** To reproduce our experiments, especially those for the 1.4 and 7B parameter models, please find the exact hyperparameters
in our paper appendix. Please find the full code of our approach for pre-training large scale LLMs in IBM's FSDP 
foundation model stack: https://github.com/foundation-model-stack/fms-fsdp/tree/reweighted-train
This code is executable out of the box. The hyperparameters in our paper appendix were used to train on 64 H100 GPUs.

## Running experiments

First you need to install the project dependencies available in the `requirements.txt` file.

Before you can train a model, you will have to download the data. 
We provide all necessary utilities for this in `src/data/`. 

To launch a training run, you may use our runner script provided under `src/run.py`. 
Please see `script/run_bigger.sh` for sample instructions.

**Note:** This project depends on Weights&Biases, which requires to have a user account with their logging service.

## Core code pieces to enable our reweighing scheme in a multi-GPU environment
Please see the appendix of our paper for more details.

```python
import torch
import torch.distributed as dist
    
# Exponential scaling function
def scale_losses(losses, r):
    return torch.exp(losses / r)

# Normalization function for losses
def normalize_losses(losses, delta=1., l_min=0., l_max=1.):
    return 2. * delta * losses / max(l_max - l_min, 1e-6) - delta * (l_max + l_min) / max(l_max - l_min, 1e-6)

# Reweighting strategies
def apply_strategy(losses, delta=1.0, strategy="linupper"):
    if strategy == "linupper": 
        return torch.minimum(losses + delta, delta * torch.ones_like(losses))
    elif strategy == "uniform": 
        # We do not reweight here. This is our baseline.
        return losses
    elif strategy == "quadratic": 
        return 1 - losses**2 / delta**2
    elif strategy == "extremes": 
        return torch.abs(losses)
    else: 
        raise NotImplementedError

# Compute batch losses from logits
def get_batch_loss_from_logits(logits, labels):
    ignore_index = -100
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    num_active = (shift_labels != ignore_index).sum(dim=1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1).long())
    return loss.view(logits.size(0), -1).sum(dim=1) / num_active, num_active

# Define reweighting function mapping
STRATEGY = "linupper"

# This has to be placed inside a training step.
...

# Assume model outputs (logits) and labels are already computed
device_losses, len_norms = get_batch_loss_from_logits(outputs, labels)

# Initialize placeholder to store the losses from all GPUs
gathered_losses = torch.zeros(
    dist.get_world_size(),
    len(device_losses),
    device=device_losses.device,
    dtype=device_losses.dtype
)

# Gather losses across all GPUs into tenor
dist.all_gather_into_tensor(gathered_losses, device_losses.detach())

# We are using a warmup phase in which the model learn a rudimentary understanding of language such that 
# the losses are actually useful for distinguishing losses. That means you should set a high r in the 
# beginning to have more uniform weights and then after the warmup start re-weighting per your requirements.
r = r_scheduler(step, cfg.initial_r)
# Compute sample weights
with torch.no_grad():
    min_loss = gathered_losses.min().item()
    max_loss = gathered_losses.max().item()
    normalized_losses = normalize_losses(gathered_losses.view(-1), delta=1., l_min=min_loss, l_max=max_loss)
    reweighted_losses = apply_strategy(normalized_losses, delta=1., strategy=STRATEGY)
    scaled_losses = scale_losses(reweighted_losses - reweighted_losses.max().item(), l=r)
    weights = scaled_losses / scaled_losses.sum()

    # for instance local_rank can be obtained with dist.get_rank()
    device_weights = weights.view(dist.get_world_size(), -1)[local_rank, :] 

# Reweight losses and scale appropriately
loss = torch.sum(device_weights * device_losses) * dist.get_world_size()
loss.backward()
```





### Attributions
This code base is based on the code of [DOGE: Domain Reweighting with Generalization Estimation](https://openreview.net/forum?id=qiKqsqwYXm). 
We thank the DOGE authors for making their code public.
