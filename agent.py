import numpy as np
import random

def epsilon_greedy_policy(state, collected_supplies, q_table, epsilon, env):
    supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(q_table[state[0]][state[1]][supply_index])

def train_agent(env, num_episodes=10000, max_steps_per_episode=100, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, min_epsilon=0.01, epsilon_decay_rate=0.001):

    q_table = np.zeros((env.size, env.size, 2 ** len(env.supply_states), 4))  
    
    for episode in range(num_episodes):
        state, collected_supplies = env.reset()
        done = False
        t = 0
        while not done and t < max_steps_per_episode:
            action = epsilon_greedy_policy(state, collected_supplies, q_table, epsilon, env)
            next_state, next_collected_supplies, reward, done = env.step(action)
            supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
            next_supply_index = int(''.join(['1' if (i, j) in next_collected_supplies else '0' for (i, j) in env.supply_states]), 2)
            q_table[state[0]][state[1]][supply_index][action] += learning_rate * \
                (reward + discount_factor * np.max(q_table[next_state[0]][next_state[1]][next_supply_index]) - q_table[state[0]][state[1]][supply_index][action])
            state, collected_supplies = next_state, next_collected_supplies
            t += 1
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay_rate))

    return q_table