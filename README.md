# Reversi-game-AI
Python project for schoolwhere the point was to make an AI for a game of reversi.

Results: Standard minmax algorithm with alpha beta pruning worked well even though the heuristics were a pain. Tried using a neural network for the heuristics but it only worked against weaker oponents. The code for the network should be fine but I think that the network was too shallow and the training data was too small.

Content:
  HAL: the main algorithm that I sent to school for evaluation and competition.
  HAL_NN: Neural network version. Includes lots of functions from the main project that are unused.
  Matrix: contains matrix functions for the neural network. Made my own so that it would easily fork on school computers
  PotentMobility: heuristic function.
  Stability: Heuristic function.
