
# AI Othello

This project implements an AI for the game of Othello using multiple neural network architectures including MLP, CNN, and LSTM. It covers the complete pipeline from data preprocessing to training and game simulation.

## Structure

- **game.py**: Main game execution script. It loads two models and simulates games between them, generating move logs and animations.
- **networks.py**: Contains implementations for the MLP, LSTM, and CNN models used to predict moves.
- **utile.py**: Utility functions for board state management, move validation, and tile flipping logic.
- **training_CNN.py**: Training script for the CNN model.
- **training_MLP.py**: Training script for the MLP model.
- **training_LSTM.py**: Training script for the LSTM model.
- **reloadModel.py**: Reloads and re-saves models for compatibility (e.g., ensuring `len_input_seq` is set).

## Dataset

- Games are stored in `.h5` files with sequences of board states and corresponding moves.
- Dataset split files: `train.txt`, `dev.txt`
- File format: each file contains a list of `.h5` game logs used for training and validation.

## Models

- **MLP**: Fully connected layers acting on flattened board states.
- **LSTM**: Processes sequences of board states to predict moves.
- **CNN**: Applies convolutions on board states to extract spatial features.

## Training

Each `training*.py` script handles loading datasets, initializing the model, and running training loops with evaluation on a validation set. Early stopping is used to prevent overfitting.

## Game Simulation

`game.py` loads two trained models and simulates two full games:
- First, Player 1 as Black vs. Player 2 as White
- Then, Player 2 as Black vs. Player 1 as White

The games are saved as GIF animations showing move progression.

## Requirements

- Python 3
- PyTorch
- NumPy, Pandas
- Matplotlib
- h5py
- tqdm
- scikit-learn

## Usage

To play a game between two trained models:

```bash
python game.py path_to_model1.pt path_to_model2.pt
```

To train models:

```bash
python training_MLP.py
python training_CNN.py
python training_LSTM.py
```
