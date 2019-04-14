import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


Experience = namedtuple("Experience", field_names="state action reward next_state done")


def tensor(array):
    return torch.tensor(array, dtype=torch.float)


class Critic(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256 + action_size, 128)
        self.qval = nn.Linear(128, 1)
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight)  # default activation is leaky relu
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.qval.weight)  # default activation is linear

    def forward(self, state, action):
        x = F.leaky_relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.qval(x).squeeze()


class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.act = nn.Linear(256, action_size)
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight)  # default non-linearity is leaky relu
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.act.weight, nn.init.calculate_gain("tanh"))

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        return torch.tanh(self.act(x))


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DDPGAgent:

    def __init__(self, state_size, action_size, load=False):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.polyak = 0.99
        self.batch_size = 128
        self.train_start = 128

        self.memory = ReplayMemory(int(1e6))

        self.actor = Actor(state_size, action_size)
        self.actor_target = Actor(state_size, action_size)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if load:
            self.load_pretrained()

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=0.0001)

    def push_experience(self, state, action, reward, next_state, done):
        self.memory.push(Experience(state, action, reward, next_state, done))

    def update_target_nets_soft(self):

        # Update critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)

        # Update actor target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)

    def take_action(self, state, act_noise=0.2):
        with torch.no_grad():
            return self.actor(state) + torch.normal(torch.zeros(self.action_size), act_noise)

    def optimize_model(self):
        if len(self.memory) < self.train_start:
            return

        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        non_final_mask = ~torch.tensor(batch.done)
        non_final_next_states = torch.stack([s for done, s in zip(batch.done, batch.next_state) if not done])

        Q_values = self.critic(state_batch, action_batch)

        #    DDPG target    #
        next_state_values = torch.zeros(self.batch_size)
        actions = self.actor_target(non_final_next_states)
        next_state_values[non_final_mask] = self.critic_target(non_final_next_states, actions)

        Q_targets = reward_batch + self.gamma * next_state_values.detach()
        #####################

        # Optimize critic
        assert Q_values.shape == Q_targets.shape
        self.critic_opt.zero_grad()
        critic_loss = F.mse_loss(Q_values, Q_targets)
        critic_loss.backward()
        self.critic_opt.step()

        # Optimize actor
        self.actor_opt.zero_grad()
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()  # Negative sign for gradient ASCENT
        actor_loss.backward()
        self.actor_opt.step()

        self.update_target_nets_soft()

    def load_pretrained(self):
        self.actor.load_state_dict(torch.load("actor_pretrained.pt"))
        self.critic.load_state_dict(torch.load("critic_pretrained.pt"))


def test_run():

    env = gym.make("BipedalWalker-v2")
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    agent = DDPGAgent(obs_size, act_size, load=True)

    state = tensor(env.reset())
    returns = 0
    while True:
        env.render()
        action = agent.take_action(state, act_noise=0.001)  # basically no noise
        next_state, reward, done, info = env.step(action.numpy())
        state = tensor(next_state)

        returns += reward

        if done:
            break

    print("Return: {:.2f}".format(returns))


def train(episodes=3000):

    env = gym.make("BipedalWalker-v2")
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    print_every = 50
    checkpoint_every = 100

    agent = DDPGAgent(obs_size, act_size)

    print("Starting training!\n")

    returns = []
    for episode in range(1, episodes + 1):

        if episode % print_every == 0:
            print("Episode        : ", episode)
            print("Memory size    : ", len(agent.memory))
            print("Best return    : {:.2f}".format(np.max(returns[-print_every:])))
            print("Avarage return : {:.2f}\n".format(np.mean(returns[-print_every:])))

        state = tensor(env.reset())
        episode_return = 0

        for step in range(1000):

            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action.numpy())
            next_state, reward = tensor(next_state), tensor(reward)

            agent.push_experience(state, action, reward, next_state, done)
            agent.optimize_model()

            episode_return += reward.item()
            if done:
                break

            state = next_state

        returns.append(episode_return)

        if episode % checkpoint_every == 0:
            print("Checkpointing the agent!\n")
            torch.save(agent.actor.state_dict(), "actor_checkpoint.pt")
            torch.save(agent.critic.state_dict(), "critic_checkpoint.pt")

    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.savefig("train.png", dpi=1000)
    plt.show()


if __name__ == "__main__":

    # test_run()
    train()


