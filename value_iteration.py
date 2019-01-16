import sys
import numpy as np


def maze_interpret(maze_file):
    with open(maze_file, 'r') as file:
        row = 0
        column = 0
        for line in file:
            line = line.rstrip('\n')  # delete '\n'
            column = len(line)
            row += 1
        maze = np.chararray((row, column))
        row = 0
    with open(maze_file, 'r') as file:
        for line in file:
            line = line.rstrip('\n')  # delete '\n'
            column = 0
            for char in line:
                maze[row][column] = char
                column += 1
            row += 1
    return row, column, maze


def value_function_trainer():
    v = np.zeros(maze.shape)
    for i in range(num_epoch):
        new_v = np.zeros(maze.shape)
        for r in range(row):
            for c in range(column):
                if maze[r][c] != b'*' and maze[r][c] != b'G':  # Don't process the obstacle and the goal
                    Q = list()
                    for action in range(num_action):  # action: go West, North, East, South
                        action_r, action_c = location_cal(action, r, c)
                        # hit the wall or hit the obstacle
                        if action_r < 0 or action_r >= row or action_c < 0 or action_c >= column or maze[action_r][
                            action_c] == b'*':
                            temp = -1 + discount_factor * v[r][c]
                        else:
                            temp = -1 + discount_factor * v[action_r][action_c]

                        Q.append(temp)
                    new_v[r][c] = max(Q)
        v = new_v
    return v


def q_value_cal_from_value_function(v):
    Q = np.zeros((num_action, row * column))
    counter = 0
    for r in range(row):
        for c in range(column):
            if maze[r][c] != b'*' and maze[r][c] != b'G':  # Don't process the obstacle and the goal
                for action in range(num_action):
                    action_r, action_c = location_cal(action, r, c)
                    if action_r < 0 or action_r >= row or action_c < 0 or action_c >= column or maze[action_r][
                        action_c] == b'*':
                        temp = -1 + discount_factor * v[r][c]
                    else:
                        temp = -1 + discount_factor * v[action_r][action_c]
                    Q[action][counter] = temp
            counter += 1
    return Q


def location_cal(direction, row, column):  # action: go West, North, East, South
    if direction == 0:  # West
        return row, column - 1
    if direction == 1:  # North
        return row - 1, column
    if direction == 2:  # East
        return row, column + 1
    if direction == 3:  # South
        return row + 1, column


if __name__ == "__main__":
    # Command line arguments
    maze_input = sys.argv[1]  # path to the environment input .txt
    value_file = sys.argv[2]  # path to output the values V(s)
    q_value_file = sys.argv[3]  # path to output the q-values Q(s,a)
    policy_file = sys.argv[4]  # path to output the optimal actions π(s)
    num_epoch = int(sys.argv[5])  # the number of epochs your program should train the agent for
    discount_factor = float(sys.argv[6])  # the discount factor γ

    # number of actions available
    num_action = 4

    row, column, maze = maze_interpret(maze_input)
    v = value_function_trainer()
    q = q_value_cal_from_value_function(v)

    with open(value_file, 'w') as output:
        for r in range(row):
            for c in range(column):
                if maze[r][c] != b'*':
                    print(str(r) + " " + str(c) + " " + str(v[r][c]), end='\n', file=output)

    with open(q_value_file, 'w') as output:
        counter = 0
        for r in range(row):
            for c in range(column):
                if maze[r][c] != b'*':
                    for action in range(num_action):
                        print(str(r) + " " + str(c) + " " + str(action) + " " + str(q[action][counter]), end='\n',
                              file=output)
                counter += 1

    with open(policy_file, 'w') as output:
        counter = 0
        for r in range(row):
            for c in range(column):
                if maze[r][c] != b'*':
                        print(str(r) + " " + str(c) + " " + str(np.argmax(q[:, counter])), end='\n', file=output)
                counter += 1
