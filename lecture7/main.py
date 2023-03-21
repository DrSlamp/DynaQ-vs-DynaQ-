import sys
import gym
import gym_environments
import numpy as np
from agent import DYNAQ
from agent1 import DYNAQplus
#choose betwen DYNAQ or DYNAQplus, change line: 12,38 & 95 for plot title
import matplotlib.pyplot as plt
import numpy as np


def run(env, agent: DYNAQplus, selection_method, episodes):
    for episode in range(episodes):
        if episode > 0:
            print(f"Episode: {episode+1}")
        observation, _ = env.reset()
        agent.start_episode()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation, selection_method)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.update(observation, action, next_observation, reward)
            observation = next_observation
        if selection_method == "epsilon-greedy":
            for _ in range(100):
                state = np.random.choice(list(agent.visited_states.keys()))
                action = np.random.choice(agent.visited_states[state])
                reward, next_state = agent.model[(state, action)]
                agent.update(state, action, next_state, reward)


if __name__ == "__main__":
    environments = ["Princess-v0", "Blocks-v0"]
    id = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    episodes = 350 if len(sys.argv) < 3 else int(sys.argv[2])

    env = gym.make(environments[id])
    agent = DYNAQplus(
        env.observation_space.n, env.action_space.n, alpha=1, gamma=0.95, epsilon=0.1
    )

    # Train
    run(env, agent, "epsilon-greedy", episodes)
    env.close()

    # Play 1
for _ in range(1): 
    env = gym.make(environments[id], render_mode="human")
    run(env, agent, "greedy", 1)
    agent.render()
   
    buff1 = 0
    buff1 = agent.step
    bufferin = [buff1]
    env.close()


    # Play 2
for _ in range(1): 
    env = gym.make(environments[id], render_mode="human")
    run(env, agent, "greedy", 1)
    agent.render()
    buff2 = 0
    buff2 = agent.step
    bufferin1 = [buff2]
    env.close()


    #inicializador

valores = [] # inicia#lizar una lista vacía
valores.append(buff1) # 
valores.append(buff2) # 

total = sum(valores) # sumar todos los valores de la lista y asignarlo a la variable total
avg = total / 2
print("steps per episode avg: ",avg) # print sum states

#plot time! 

# x-axis values
x = [0,episodes]
# y-axis values
y = [0,avg]
  
# plotting points as a scatter plot
plt.scatter(x, y, label= "stars", color= "orange", 
            marker= "*", s=30)
  
# x-axis label
plt.xlabel('x - episodes')
# frequency label
plt.ylabel('y - average steps per episode')
# plot title
plt.title('DYNAQplus (2Run)')
# showing legend
plt.legend()
  
# function to show the plot
plt.show()