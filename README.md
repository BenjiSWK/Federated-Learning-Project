# Federated Learning and Privacy in AI: Convergence & Vulnerabilities

The project investigates two fundamental aspects of Federated Learning (FL): 
1. **Algorithmic Efficiency:** The mathematical and empirical convergence of the **FedAvg** algorithm on highly heterogeneous (Non-IID) data.
2. **Data Privacy:** The critical vulnerabilities of gradient sharing, demonstrated through **Gradient Inversion** and **Label Leakage from Gradients (LLG)** attacks.

## Core Implementations

All experiments are conducted using a **Regularized Multinomial Logistic Regression** model on the **MNIST** dataset.

### Part 1: Convergence and Client Drift
* Implementation of the standard **FedAvg** and **FedSGD** protocols.
* Custom data distribution algorithms to simulate extreme **Non-IID (heterogeneous)** environments using a shard-based approach.
* Analysis of hyperparameters (Local Epochs $E$, Batch Size $B$, Participation Fraction $C$, and Decaying Learning Rate $\eta_t$) and their impact on the *Client Drift* phenomenon.

### Part 2: Privacy Attacks
* **Exact Gradient Inversion:** Mathematical demonstration and visual reconstruction of private input images ($\tilde{x}$) exploiting the collinearity in the local weight updates.
* **Label Leakage from Gradients (LLG):** Implementation of the LLG attack, analyzing the negative gradients to deterministically extract the exact distribution of local labels, even after the global model has converged.

## How to Run

### Prerequisites
Ensure you have the required Python libraries installed (e.g., PyTorch, NumPy, Matplotlib).

```bash
pip install torch torchvision numpy matplotlib


