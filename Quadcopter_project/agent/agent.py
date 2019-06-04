# TODO: your agent here!
import numpy as np
import random
from task import Task
from collections import namedtuple, deque

from agents.myactor import MyActor
from agents.mycritic import MyCritic

class MyStore:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, new_state, is_done):
        exp = self.experience(state, action, reward, new_state, is_done)
        self.memory.append(exp)

    def sample(self, batch_size=60):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)


class NoiseDetector:
    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class MyAgent():
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        self.actor_local = MyActor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = MyActor(self.state_size, self.action_size, self.action_low, self.action_high)

        self.critic_local = MyCritic(self.state_size, self.action_size)
        self.critic_target = MyCritic(self.state_size, self.action_size)

        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Handling noise
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.3
        self.noise = NoiseDetector(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Handling memory
        self.buffer_size = 1000000
        self.batch_size = 60
        self.memory = MyStore(self.buffer_size, self.batch_size)

        # Hyperparameters
        self.gamma = 0.98  # discount factor
        self.tau = 0.002  # for soft update of target parameters
        
        # Scoring parameters
        self.best_score = -np.inf
        self.score = 0

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.score = 0
        return state

    def step(self, action, reward, next_state, done):
        # add to memory
        self.memory.add(self.last_state, action, reward, next_state, done)

        # start to learn once memory samples are enough
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        self.last_state = next_state
        
        # Update score and best_score
        self.score += reward
        if done:
            if self.score > self.best_score:
                self.best_score = self.score

    def act(self, states):
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)