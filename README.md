# **Project: Build Large Language Models From Scratch**

A step-by-step project series focused on understanding how large language models are built from the ground up, from data preprocessing and attention to pre-training, fine-tuning, and evaluation. The goal of the project is to train the reader to think like a fundamental machine learning engineer rather than only using ready-made LLM APIs.

## Project Overview

This project walks through the complete lifecycle of building an LLM from scratch. It begins with the foundations of text processing and tokenization, moves into the internal mechanics of attention and Transformer blocks, and then progresses to training, fine-tuning, and evaluation. The codebase is designed in a modular way so each building block can be understood, modified, and extended independently.

## Motivation

The main purpose of the project is educational: to make the inner workings of LLMs transparent and accessible. Instead of treating an LLM as a black box, the project emphasizes the “nuts and bolts” of model construction, helping learners develop a deeper intuition for how language models learn, represent context, and generate predictions.

## What This Project Covers

The lecture series and codebase cover three major stages:

1. Foundation stage.
2. Pre-training stage.
3. Fine-tuning and evaluation stage.

Each stage builds on the previous one, similar to constructing a house from the foundation upward.

## Stage 1: Foundation

This stage focuses on the core components required before training begins.

### Data Preprocessing

The first step is converting raw text into a form that a neural network can use. Text is broken into tokens, tokens are mapped to token IDs, and those IDs are converted into dense vector representations called embeddings. Positional embeddings are added so the model can preserve token order, producing the final input embeddings used by the model.

### Intuition

Token embeddings capture meaning, while positional embeddings capture order. Both are necessary because language depends not only on what words appear, but also on where they appear in the sequence.

### Attention Mechanism

Attention is the central idea that makes modern LLMs powerful. Instead of processing each token in isolation, the model computes relationships between tokens using queries, keys, and values. Attention scores are scaled, masked causally, normalized with softmax, and used to produce context vectors.

### Intuition

Attention tells the model which other tokens matter most when interpreting a given token. This is what allows the model to build contextual understanding rather than static word meanings.

### Multi-Head Attention

Multiple attention heads are used in parallel so the model can learn different kinds of relationships at the same time. One head may focus on syntax, another on semantic relations, and another on long-range dependencies.

### Transformer Architecture

The Transformer block combines:

- Layer normalization.
- Multi-head attention.
- Residual connections.
- Feedforward neural networks.

These blocks are stacked many times to form the full LLM architecture.

### Intuition

Attention helps tokens communicate, while the feedforward layers help the model transform and refine the information it has gathered.

## Stage 2: Pre-Training

This stage explains how the model learns language patterns from large amounts of data.

### Next-Token Prediction

The model is trained to predict the next token in a sequence. Given an input sentence, the LLM outputs logits over the vocabulary, which are converted into probabilities for the next token.

### Loss Function

The training objective is typically cross-entropy loss, which measures the difference between the predicted next token and the actual next token.

### Backpropagation

After computing the loss, gradients are calculated for all trainable parameters, including:

- Token embeddings.
- Positional embeddings.
- Query, key, and value matrices.
- LayerNorm parameters.
- Feedforward network weights.
- Final output layers.


### Optimization

The model updates parameters using gradient-based optimization methods such as Adam or AdamW. This iterative process is repeated over many batches and epochs.

### Intuition

Pre-training teaches the model general language structure by repeatedly correcting its next-token predictions.

### Practical Note

The lecture series demonstrates this process on a small dataset and a smaller setup, while real-world LLM pre-training requires massive compute and large-scale datasets.

## Stage 3: Fine-Tuning

After pre-training, the model is adapted for specific downstream tasks.

### Classification Fine-Tuning

One project trains an LLM for email classification, such as distinguishing spam from non-spam messages.

### Intuition

The model already understands language broadly from pre-training, and fine-tuning teaches it how to solve a specific task.

### Instruction Fine-Tuning

Another project trains the model to follow instructions, such as converting active voice to passive voice or responding to user prompts in a structured way.

### Intuition

Instruction tuning makes the model more useful as a general assistant by teaching it how to respond to commands.

## Evaluation

The project also covers how to measure model quality.

### Benchmark Evaluation

One approach uses benchmark-style testing such as MMLU, which evaluates performance across many different tasks.

### Human Evaluation

Another approach is human judgment, where people compare outputs and rate model quality.

### LLM-as-Judge

The project also demonstrates using a stronger LLM to evaluate another model’s response by comparing the predicted output against the target output and assigning a score.

### Intuition

Evaluation is not just about correctness. For language models, it also includes usefulness, coherence, and instruction-following quality.

## Key Learnings

This project teaches how to:

- Build a tokenization and embedding pipeline.
- Implement attention from scratch.
- Assemble Transformer blocks into a full architecture.
- Train a next-token prediction model.
- Fine-tune the model for classification and instruction-following.
- Evaluate model outputs using benchmarks, humans, and LLM judges.


## Why This Project Matters

The project helps learners move beyond using LLMs as black-box services. It develops a working understanding of how language models are engineered, trained, and adapted, which is valuable for research, model development, and advanced ML engineering.

## Suggested Extensions

The codebase is intentionally structured so it can be extended for research and experimentation. Possible next steps include:

- Changing the number of Transformer blocks.
- Trying different learning rates and optimizers.
- Comparing evaluation methods.
- Exploring smaller or more efficient model variants.
- Studying the effect of architectural changes on performance.


## Outcome

By completing the series, the learner gains hands-on experience in building:

- A next-token prediction LLM from scratch.
- A spam classification LLM.
- An instruction-tuned assistant model.

This makes the project a strong portfolio piece for interviews, research discussions, and practical ML/LLM engineering work.

