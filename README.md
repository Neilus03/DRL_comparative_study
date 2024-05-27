# Breakout Reinforcement Learning Implementations

Welcome to the Breakout Reinforcement Learning Implementations repository. This project includes various implementations of Reinforcement Learning algorithms applied to the classic Breakout game environment. It aims to provide reproducible results for the research presented in our paper _A COMPARATIVE STUDY OF DEEP REINFORCEMENT
LEARNING MODELS: DQN VS PPO VS A2C_.

## 🔩 Directory Structure

The repository is organized into the following directories, each containing a specific approach to solving Breakout using different RL algorithms:

- **`BreakOut_base`**: 
  - *Description*: Base implementation of the Deep Q-Network (DQN) for the Breakout game.

- **`BreakOut_sb3_A2C`**: 
  - Implementation of the Advantage Actor-Critic (A2C) algorithm using Stable Baselines 3.

- **`BreakOut_sb3_PPO`**: 
  - Proximal Policy Optimization (PPO) approach using Stable Baselines 3.
- **`Breakout_sb3_DQN`**: 
  - Implementation of DQN using Stable Baselines 3.
    
*Details*: The README in this folder will guide you through the necessary steps for training and testing.

## 📦 Installation

Ensure Python 3.x is installed along with the following dependencies, which are common across all implementations:

```bash
pip install -r requirements.txt
```

## 🎮 Usage

1. **Clone & Set Up** this repository:
    ```bash
    git clone https://github.com/Neilus03/DRL_comparative_study
    cd DRL_comparative_study
    pip install -r requirements.txt
    ```

2. **Navigate** to the desired implementation directory, e.g., for PPO:
    ```bash
    cd BreakOut_sb3_PPO
    ```
    
3. **Configure** the `config.py` file to adjust the model training, the wandb account, and the saving model options.

4. **Execute** the `train.py` to run the model or `test.py` if you already have a pre-trained model to test.

## 👥 Contributing

We welcome contributions! If you have suggestions or improvements, feel free to create a pull request or open an issue.

## 📧 Contact

For any questions or inquiries, please reach out to:

[Daniel Vidal](https://www.linkedin.com/in/daniel-alejandro-vidal-guerra-21386b266/)

[Neil de la Fuente](https://www.linkedin.com/in/neil-de-la-fuente/)

---

Thank you for visiting! We hope you find this repository useful for reproducing the results presented in our paper.

---
