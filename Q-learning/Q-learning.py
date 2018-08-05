import gym
import numpy as np
import random

''' Q(quality)-learning is a value-based Reinforcement Learning algorithm.
The Q-value is the estimated maximum future reward, for each action and state.
We can create a Q-table to store these values, and typically take the best
action in that given state. We keep updating this Q-table until we have an
accurate table for our situation. We initialize these values to 0 to start out.'''

env = gym.make('Taxi-v2')

action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))
print(qtable)
# HYPER PARAMETERS
total_episodes = 10000
learning_rate = 0.8
max_steps = 99
gamma = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

rewards = []
for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_reward = 0

    for step in range(max_steps):
        e_e_tradeoff = random.uniform(0,1)
        # take a exploit step
        if e_e_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        # take a random explote step
        else:
            action = env.action_space.sample()

        new_state, reward, done, _ = env.step(action)
        # update our q-table
        qtable[state, action] = qtable[state, action] + learning_rate * \
                            (reward + gamma * np.max(qtable[new_state,:]) - qtable[state,action])

        total_reward += reward

        state = new_state

        if done == True:
            break

    episode += 1
    # reduce epsilon to exploit more than exploit after we learn more
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    rewards.append(total_reward)

print ("Score over time: " +  str(sum(rewards)/total_episodes))
print(qtable)


env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    total_reward = 0
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])

        new_state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(total_reward)
            break
        state = new_state

env.close()
