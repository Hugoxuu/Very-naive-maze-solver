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


if __name__ == "__main__":
    # Command line arguments
    maze_input = sys.argv[1]  # path to the environment input .txt
    output_file = sys.argv[2]  # path to output feedback from the environment after the agent takes sequence of actions
    action_seq_file = sys.argv[3]  # path to the file containing a sequence of actions in order

    maze_env = Environment(maze_input)
    with open(action_seq_file, 'r') as input, open(output_file, 'w') as output:
        line = input.readline()
        line = line.rstrip('\n')  # delete '\n'
        action_seq = line.split(' ')
        for action in action_seq:
            next_x, next_y, reward, is_terminal = maze_env.step(int(action))
            print(str(next_x) + ' ' + str(next_y) + ' ' + str(reward) + ' ' + str(is_terminal), end='\n', file=output)
