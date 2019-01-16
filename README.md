# Very-naive-maze-solver
This program implements reinforcement learning algorithms for solving a maze.

1. "value_iteration.py" implements Value-Iteration method for solving the maze, i.e. this program knows the maze fully at the first place.\
For this program, you shall specify the number of epoch, and discount factor.

2. "q_learning.py" implements Q-learning method for solving the maze, i.e. this program interacts with the maze environment and finds the optimized path gradually. ("envrionment.py" is the class for interaction)\
For this program, you shall specify the number of episodes, max episode length, learning rate, discount factor, and epsilon, respectively.

3. A very simple maze is given, named "medium_maze.txt".
