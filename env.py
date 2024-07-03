import numpy as np
import matplotlib.pyplot as plt

class ReinforcementLearning:
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start_state = (0, 0)
        self.goal_state = (size-1, size-1)
        self.zombie_states = [(1, 1), (2, 3), (3, 0), (3, 4)]
        self.supply_states = [(0, 4), (2, 1), (4, 0)]
        self.stone_states = [(1, 2), (3, 3)]
        self.supplies_collected = set()
        for i, j in self.zombie_states:
            self.grid[i][j] = 1
        for i, j in self.supply_states:
            self.grid[i][j] = 2
        for i, j in self.stone_states: 
            self.grid[i][j] = 3
    
    def reset(self):
        self.current_state = self.start_state
        self.supplies_collected = set()
        return self.current_state, tuple(self.supplies_collected)
    
    def step(self, action):
        i, j = self.current_state

        if action == 0: 
            next_state = (max(i-1, 0), j)
        elif action == 1: 
            next_state = (min(i+1, self.size-1), j)
        elif action == 2:  
            next_state = (i, max(j-1, 0))
        elif action == 3:  
            next_state = (i, min(j+1, self.size-1))

        if self.grid[next_state[0]][next_state[1]] != 3:
            self.current_state = next_state
        if self.current_state == self.goal_state:
            if len(self.supplies_collected) == len(self.supply_states):
                reward = 10 
                done = True
            else:
                reward = -1 
                done = False
        elif self.current_state in self.zombie_states:
            reward = -5  
            done = True
        elif self.current_state in self.supply_states and self.current_state not in self.supplies_collected:
            self.supplies_collected.add(self.current_state)
            reward = 2  
            done = False
        else:
            reward = -0.1  
            done = False

        return self.current_state, tuple(self.supplies_collected), reward, done

    def render(self):
        print('\n')
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.current_state:
                    print('A', end=' ')  
                elif self.grid[i][j] == 0:
                    if (i, j) == self.goal_state:
                        print('G', end=' ')  
                    else:
                        print('.', end=' ')
                elif self.grid[i][j] == 1:
                    print('X', end=' ')  
                elif self.grid[i][j] == 2:
                    if (i, j) in self.supplies_collected:
                        print('.', end=' ')
                    else:
                        print('O', end=' ')  
                elif self.grid[i][j] == 3:
                    print('S', end=' ')  
            print()
        print()
    
    def render_graph(self):
        if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots()
            plt.ion()
    
        self.ax.clear() 
        
        for i in range(self.size):
            for j in range(self.size):
                color = 'white'  
                if (i, j) == self.current_state:
                    color = 'blue'  
                elif self.grid[i][j] == 0:
                    if (i, j) == self.goal_state:
                        color = 'green'  
                elif self.grid[i][j] == 1:
                    color = 'red'  
                elif self.grid[i][j] == 2:
                    if (i, j) in self.supplies_collected:
                        color = 'white'  
                    else:
                        color = 'yellow'  
                elif self.grid[i][j] == 3:
                    color = 'gray'  

                adjusted_i = self.size - 1 - i

                self.ax.add_patch(plt.Rectangle((j, adjusted_i), 1, 1, color=color))

        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_xticks(np.arange(0, self.size, 1))
        self.ax.set_yticks(np.arange(0, self.size, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(True)
        
        plt.draw()
        plt.pause(0.5)

    def show_q_table(self, q_table):
        print('-----------------------------------------------------------------')
        print('Q-Table:')
        print('-----------------------------------------------------------------')
        num_supplies = len(self.supply_states)
        for supply_index in range(2 ** num_supplies):
            collected_supplies = format(supply_index, f'0{num_supplies}b')
            print(f'Collected Supplies: {collected_supplies}')
            for i in range(self.size):
                for j in range(self.size):
                    if self.grid[i][j] == 0 or self.grid[i][j] == 2:
                        for action in range(4):
                            print('%.2f' % q_table[i][j][supply_index][action], end='\t')
                        print()
                    else:
                        print('NULL', end='\t' * 4)
                    print()
                print()
            print()

    def show_policy(self, q_table):
        print('\n Policy:')
        num_supplies = len(self.supply_states)
        for supply_index in range(2 ** num_supplies):
            collected_supplies = format(supply_index, f'0{num_supplies}b')
            print(f'Collected Supplies: {collected_supplies}')
            for i in range(self.size):
                for j in range(self.size):
                    if self.grid[i][j] == 0 or self.grid[i][j] == 2:
                        action = np.argmax(q_table[i][j][supply_index])
                        if action == 0:
                            print('UP', end=' ')
                        elif action == 1:
                            print('DOWN', end=' ')
                        elif action == 2:
                            print('LEFT', end=' ')
                        elif action == 3:
                            print('RIGHT', end=' ')
                    else:
                        print('STAY', end=' ')
                print()
            print()