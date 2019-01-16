import sys
import numpy as np


class Environment(object):

    def __init__(self, input):

        # get the shape of maze
        with open(input, 'r') as file:
            self.row = 0
            self.column = 0
            for line in file:
                line = line.rstrip('\n')  # delete '\n'
                self.column = len(line)
                self.row += 1

        # current state coordinates
        self.x = self.row - 1
        self.y = 0

        # extract info to maze array
        self.maze = np.chararray((self.row, self.column))
        with open(input, 'r') as file:
            row = 0
            for line in file:
                line = line.rstrip('\n')  # delete '\n'
                column = 0
                for char in line:
                    self.maze[row][column] = char
                    column += 1
                row += 1

    # return next_x, next_y, reward, is_terminal
    # set the current state to the next state
    def step(self, action):

        # already terminal state
        if self.is_terminal(self.maze, self.x, self.y) == 1:
            return self.x, self.y, 0, 1

        next_x, next_y = self.location_cal(action, self.x, self.y)
        # The case of not hitting the obstacle or the wall
        if 0 <= next_x < self.row and 0 <= next_y < self.column and self.maze[next_x][next_y] != b'*':
            # update the current state
            self.x = next_x
            self.y = next_y

        is_terminal = self.is_terminal(self.maze, self.x, self.y)

        return self.x, self.y, -1, is_terminal

    def reset(self):
        self.x = self.row - 1
        self.y = 0
        return self.x, self.y

    @staticmethod
    def location_cal(direction, row, column):  # action: go West, North, East, South
        if direction == 0:  # West
            return row, column - 1
        if direction == 1:  # North
            return row - 1, column
        if direction == 2:  # East
            return row, column + 1
        if direction == 3:  # South
            return row + 1, column

    @staticmethod
    def is_terminal(maze, x, y):
        if maze[x][y] == b'G':
            return 1
        else:
            return 0


def q_value_trainer(maze):
    q = np.zeros((maze.row, maze.column, num_action))
    for episode in range(num_episodes):
        cur_x, cur_y = maze.reset()
        for i in range(max_episode_length):
            best_action = np.argmax(q[cur_x][cur_y])
            action = action_generator(best_action)  # Generating actions
            old_q = q[cur_x][cur_y][action]

            next_x, next_y, reward, is_terminal = maze.step(action)  # Interact with the environment with action

            # Calculate the new q value for that specified action in the current location
            new_q = (1 - learning_rate) * old_q + learning_rate * (reward + discount_factor * max(q[next_x][next_y]))

            # Update q and current location
            q[cur_x][cur_y][action] = new_q
            cur_x = next_x
            cur_y = next_y

            # If reaching goal state, stop
            if is_terminal == 1:
                break

    return q


def action_generator(best_action):
    rand_num = np.random.rand()
    if rand_num > epsilon:  # prob: 1-e
        return best_action
    else:
        return np.random.choice([0, 1, 2, 3])


if __name__ == "__main__":
    # Command line arguments
    maze_input = sys.argv[1]  # path to the environment input.txt
    value_file = sys.argv[2]  # path to output the values V(s)
    q_value_file = sys.argv[3]  # path to output the q-values Q(s,a)
    policy_file = sys.argv[4]  # path to output the optimal actions π(s)
    num_episodes = int(sys.argv[5])  # the number of episodes your program should train the agent for
    max_episode_length = int(sys.argv[6])  # the maximum of the length of an episode
    learning_rate = float(sys.argv[7])  # the learning rate α of the q learning algorithm
    discount_factor = float(sys.argv[8])  # the discount factor γ
    epsilon = float(sys.argv[9])  # the value e for the epsilon-greedy strategy

    # number of actions available
    num_action = 4

    env = Environment(maze_input)
    q = q_value_trainer(env)

    with open(q_value_file, 'w') as output:
        for r in range(env.row):
            for c in range(env.column):
                if env.maze[r][c] != b'*':
                    for action in range(num_action):
                        print(str(r) + " " + str(c) + " " + str(action) + " " + str(q[r][c][action]), end='\n',
                              file=output)

    with open(value_file, 'w') as value, open(policy_file, 'w') as policy:
        for r in range(env.row):
            for c in range(env.column):
                if env.maze[r][c] != b'*':
                    print(str(r) + " " + str(c) + " " + str(max(q[r][c])), end='\n', file=value)
                    print(str(r) + " " + str(c) + " " + str(np.argmax(q[r][c])), end='\n', file=policy)
