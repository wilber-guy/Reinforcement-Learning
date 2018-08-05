import gym
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from skimage.color import rgb2gray # Help us to gray our frames
# Preprocessing to reduce to grey scale
def preprocess_frame(frame):
    grey = rgb2gray(frame)
    final = grey /255.0
    return final


class Agent():
    def __init__(self, state_size, action_size, game):
        self.weight_backup      = game + "_weight.h5"
        self.state_size         = state_size       # (210, 160, 3)
        self.action_size        = action_size
        # deque = Double Ended Que, removes oldest item with the newest
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        model.add(Conv2D(input_shape=(self.state_size[0], self.state_size[1],1),
                                            filters=32,
                                            kernel_size=[8,8],
                                            strides=[4,4],
                                            activation='relu'))
        model.add(Conv2D(filters=64,
                                            kernel_size=[8,8],
                                            strides=[4,4],
                                            activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()

        if os.path.isfile(self.weight_backup):

            model.load_weights(self.weight_backup)
            #self.exploration_rate = self.exploration_min

        return model

    def save_model(self):
            self.brain.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            print(next_state.size)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            print(state.size)
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class Game:
    def __init__(self, game):
        self.sample_batch_size = 32
        self.episodes          = 100
        self.env               = gym.make(game)

        self.state_size        = self.env.observation_space.shape
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size, game)


    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = preprocess_frame(state)
                state = np.array((None,state[0], state[1],1))
                #print(state)
                #print('#######', state.ndim, state.shape)
                done = False
                index = 0

                while not done:
                    # RENDER EVERY 50 games
                    if index_episode % 10 == 0:
                        self.env.render()

                    action = self.agent.act(state)
                    #print(action)

                    next_state, reward, done, _ = self.env.step(action)

                    next_state = preprocess_frame(next_state)
                    next_state = np.array((None, next_state[0],next_state[1],1))

                    #print('### ASS ### \n' *20)
                    #print(state, action, reward, next_state, done, '### ASS2 ###')
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                print("Episode {}# Score: {}".format(index_episode, index+1))
                #self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()

if __name__ == "__main__":
    game = Game('Boxing-v0')
    game.run()
