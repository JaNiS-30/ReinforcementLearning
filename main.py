from env import ReinforcementLearning  
from agent import train_agent
import numpy as np


print("Escolha uma opção:\n1 - Treinar um novo agente\n2 - Carregar tabela Q existente")
choice = input("Digite sua escolha (1 ou 2): ")
while choice not in ['1', '2']:
    print("Entrada inválida. Por favor, escolha '1' para treinar um novo agente ou '2' para carregar tabela Q existente.")
    choice = input("Digite sua escolha (1 ou 2): ").strip().lower()

env = ReinforcementLearning()

if choice == '1':
    print("Treinando...")
    q_table = train_agent(env, num_episodes=10000, max_steps_per_episode=100, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, min_epsilon=0.01, epsilon_decay_rate=0.001)
    np.save('q_table.npy', q_table)
    print("Treinamento concluído. q_table salva como 'q_table.npy'.")
elif choice == '2':
    q_table_path = input("Digite o caminho da tabela Q a ser carregada: ")
    q_table = np.load(q_table_path)
    print(f"q_table carregada de '{q_table_path}'.")
else:
    print("Escolha inválida. Por favor, digite 1 para treinar ou 2 para carregar.")


print("Escolha o modo de visualização:")
print("[g] Gráfico")
print("[t] Terminal")
escolha = input("Digite 'g' para gráfico ou 't' para terminal: ").strip().lower()

while escolha not in ['g', 't']:
    print("Entrada inválida. Por favor, escolha 'g' para gráfico ou 't' para terminal.")
    escolha = input("Digite 'g' para gráfico ou 't' para terminal: ").strip().lower()

# Test agent
state, collected_supplies = env.reset()
done = False
while not done:
    supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
    action = np.argmax(q_table[state[0]][state[1]][supply_index])
    next_state, next_collected_supplies, reward, done = env.step(action)

    if escolha == 'g':
        env.render_graph()
    elif escolha == 't':
        env.render()

    state, collected_supplies = next_state, next_collected_supplies