# CS285 Deep Reinforcement Learning

This repository documents my implementations, experiments, notes, and reports
for Berkeley CS285: Deep Reinforcement Learning, Decision Making, and Control.

## Repository Structure

- `hw1`: Behavior Cloning and DAgger
- `hw2`: Policy Gradients
- `hw3`: DQN and Soft Actor-Critic
- `hw4`: Model-Based Reinforcement Learning
- `hw5`: Exploration, Offline RL, and Online Fine-Tuning

## Reproducibility Notice

The dependency versions in this repository are primarily intended to reproduce
the original CS285 Fall 2023 coursework environment. Some packages are
intentionally pinned to older versions and should not be treated as recommended
production dependencies.

Use a separate Conda or virtual environment for each homework directory.
Because each homework package is named `cs285`, avoid installing multiple
homework packages into the same Python environment.

## Security Notes

Only load trusted `.pkl` or other pickle-based expert policy and dataset files.
Python pickle deserialization may execute arbitrary code.

Do not commit API keys, passwords, private keys, `.env` files, local server
paths, training logs, checkpoints, or private datasets.

Installation scripts may install system packages and recreate Conda
environments. Read each script before executing it.

## Academic Integrity

This repository records my personal implementation and learning process.
Please follow your institution's academic integrity policy. Do not submit this
work, reports, or experimental results as your own coursework.

## Course

Berkeley CS285: Deep Reinforcement Learning, Decision Making, and Control.
