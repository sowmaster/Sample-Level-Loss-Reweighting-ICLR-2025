# Loss-based sample-level reweighting for LLM pre-training

This repository contains the codebase to replicate all experiment of the paper titled: Dynamic Loss-Based Sample Reweighting for Improved Large Language Model Pretraining. 

## Experiment configurations
You may find all configuration files in the `config/` folder of this repository.

## Running experiments

First you need to install the project dependencies available in the `requirements.txt` file.

Before you can train a model, you will have to download the data. 
We provide all necessary utilities for this in `src/data/`. 

To launch a training run, you may use our runner script provided under `src/run.py`. 
Please see `script/run_bigger.sh` for sample instructions.

**Note:** This project depends on Weights&Biases, which requires to have a user account with their logging service.