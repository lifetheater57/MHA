# Modular Hierarchical Analysis Implementation (Score Matching and EM)

Implementation of method from https://arxiv.org/abs/1805.09567

Modular Hierarchical Analysis (MHA) implemented using two algorithms:
- Score Matching: The parameter updates were taken from the paper.
- Expectation-Maximization (EM): The parameters were derived by us.

To try the method, create a virtual environment with the requirements specified in `requirements.txt`. In `experiment.py`, we test MHA (EM and Score Matching) against factor analysis on synthetic data.
