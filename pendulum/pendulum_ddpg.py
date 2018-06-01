import sys
import gym
import torch
import pylab
import random
import argparse
import numpy as np
from collections import deque
from datetime import datetime
from copy import deepcopy
from skimage.transform import resize
from skimage.color import rgb2gray
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py

class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min,
                                                       n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Actor(nn.Module):
    def __init__(self, obs_size, action_size, action_range):
        self.action_range = action_range
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


class Critic(nn.Module):
    def __init__(self, obs_size, action_size, action_range):
        self.action_range = action_range
        super(Critic, self).__init__()
        self.before_action = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU()
        )
        self.after_action = nn.Sequential(
            nn.Linear(400 + action_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, action):
        x = self.before_action(x)
        x = torch.cat([x, action], dim=2)
        x = self.after_action(x)
        return x


class DDPG(object):
    def __init__(self, options):
        # hyperparameter
        self.memory_size = options.get('memory_size', 1000000)
        self.action_size = options.get('action_size')
        self.action_range = options.get('action_range')
        self.obs_size = options.get('obs_size')
        self.batch_size = options.get('batch_size')
        self.actor_lr = options.get('actor_lr')
        self.critic_lr = options.get('critic_lr')
        self.gamma = options.get('gamma')
        self.decay = options.get('decay')
        self.tau = options.get('tau')

        # actor model
        self.actor = Actor(self.obs_size, self.action_size, self.action_range)
        self.actor_target = Actor(self.obs_size, self.action_size, self.action_range)

        # critic model
        self.critic = Critic(self.obs_size, self.action_size, self.action_range)
        self.critic_target = Critic(self.obs_size, self.action_size, self.action_range)

        # memory(uniformly)
        self.memory = deque(maxlen=self.memory_size)

        # explortion
        self.ou = OrnsteinUhlenbeckProcess(args.ou_theta, sigma=args.ou_sigma, mu=args.ou_mu)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=self.decay)

        # initialize target model
        self.update_target_model()
        self.ou.reset_states()

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        model_action = self.actor(state).detach().numpy() * self.action_range
        action = model_action + self.ou.sample()
        return action

    def update_target_model(self):
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((deepcopy(state), action, reward, deepcopy(next_state), done))

    def _get_sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def train(self):
        minibatch = np.array(self._get_sample(self.batch_size)).transpose()

        states = np.stack(minibatch[0], axis=0)
        actions = np.stack(minibatch[1], axis=0)
        rewards = np.stack(minibatch[2],axis=0)
        next_states = np.stack(minibatch[3], axis=0)
        dones = minibatch[4].astype(int)

        rewards = torch.Tensor(rewards)
        dones = torch.Tensor(dones)
        actions = torch.Tensor(actions)

        # critic update
        self.critic_optimizer.zero_grad()
        states = torch.Tensor(states)
        next_states = torch.Tensor(next_states)
        next_actions = self.actor_target(next_states)

        pred = self.critic(states, actions)
        next_pred = self.critic_target(next_states, next_actions)

        target = rewards + (1-dones) * self.gamma * next_pred.view(-1)
        critic_loss = F.mse_loss(pred.view(-1), target)
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        self.actor_optimizer.zero_grad()
        actor_loss = self.critic(states, actions).mean()
        actor_loss = -actor_loss
        actor_loss.backward()
        self.actor_optimizer.step()


def main(args):
    env = gym.make(args.env)

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    print(action_size, action_range)

    args_dict = vars(args)
    args_dict['action_size'] = action_size
    args_dict['obs_size'] = obs_size
    args_dict['action_range'] = action_range

    scores, episodes = [], []
    agent = DDPG(args_dict)
    recent_reward = deque(maxlen=100)
    frame = 0

    for e in range(args.episode):
        done = False
        score = 0
        step = 0
        done = False
        state = env.reset()
        state = np.reshape(state, [1, agent.obs_size])
        while not done:
            step += 1
            frame += 1
            if args.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)

            next_state, reward, done, info = env.step([action])
            next_state = np.reshape(next_state, [1,agent.obs_size])

            reward = float(reward[0,0])
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)

            score += reward
            state = next_state
            if frame > agent.batch_size:
                agent.train()
                agent.update_target_model()

            if frame % 2000 == 0:
                print('now time : ', datetime.now())
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/pendulum_ddpg.png")

            if done:
                recent_reward.append(score)
                # every episode, plot the play time
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "   steps:", step,
                      "    recent reward:", np.mean(recent_reward))

                # if the mean of scores of last 10 episode is bigger than 400
                # stop training


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPG on pytorch')

    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--episode', default=10000, type=int, help='the number of episode')
    parser.add_argument('--render', default=False, type=bool, help='is render')
    parser.add_argument('--memory_size', default=500000, type=int, help='replay memory size')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--actor_lr', default=1e-4, type=float, help='')
    parser.add_argument('--critic_lr', default=1e-3, type=float, help='minibatch size')
    parser.add_argument('--gamma', default=0.99, type=float, help='discounted factor')
    parser.add_argument('--decay', default=1e-2, type=int, help='critic weight decay')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')

    args = parser.parse_args()
    print(vars(args))
    main(args)
