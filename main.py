n_episodes = 300
max_t = 1000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.985
BETA = 0.025

import matplotlib.pyplot as plt
# %matplotlib inline

import sys
import numpy as np
import pandas as pd
import torch
from collections import deque
import matplotlib.pyplot as plt
from agent import Agent

from mlagents.envs import UnityEnvironment


print("Python version:")
print(sys.version)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

# ------------------ FOR WINDOWS ------------------ #
# env = UnityEnvironment(file_name = "pyramid_window/Unity Environment.exe")
# ------------------ FOR LINUX -------------------- #
env = UnityEnvironment(file_name = "pyramid_linux/pyramid.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

state_size = env_info.vector_observations.shape[1]
action_size = brain.vector_action_space_size[0]
num_agents = len(env_info.agents)

agent = Agent(state_size, action_size, seed = 0)
scores = []                        
scores_window = deque(maxlen=50)   
eps = eps_start 
done = False

for i_episode in range(n_episodes):
    env_info = env.reset(train_mode= True)[brain_name]
    states = env_info.vector_observations
    agent_score = np.zeros(num_agents)
    done = False
    for t in range(max_t):
        actions, r_int = agent.parallel_act(states, eps)
        # print(r_int)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations 
        rewards = env_info.rewards                 
        dones = env_info.local_done
        r = np.asarray(rewards) + BETA * np.asarray(r_int)   
        agent.step(states, actions, r, next_states, dones)
        states = next_states
        agent_score += np.asarray(rewards)
        if done:
            break

    scores_window.append(np.mean(agent_score))
    scores.append(np.mean(agent_score))
    eps = max(eps_end, eps_decay*eps) # decrease epsilon
    # print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

    if i_episode % 10 == 0 or i_episode <= 5:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(agent_score)))
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

    if np.mean(agent_score) >= 1.7:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(agent_score)))
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        break

# Save figure
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode Number')
plt.savefig('Results.png')
# Save data
df = pd.DataFrame(scores)
df.to_csv('score.csv', index=False)
print ("figure + data saved")
    