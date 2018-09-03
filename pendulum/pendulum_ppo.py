import gym
import numpy as np
import random

import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import torch.nn as nn
import torch

from collections import deque

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def make_batch(sample, agent):
    sample = np.stack(sample)
    discounted_return = np.empty([NUM_STEP, 1])

    s = np.reshape(np.stack(sample[:, 0]), [NUM_STEP, agent.input_size])
    s1 = np.reshape(np.stack(sample[:, 3]), [NUM_STEP, agent.input_size])
    y = sample[:, 1]
    r = np.reshape(np.stack(sample[:, 2]), [NUM_STEP, 1])
    d = np.reshape(np.stack(sample[:, 4]), [NUM_STEP, 1])
    with torch.no_grad():
        state = torch.from_numpy(s)
        state = state.float()
        _, _, _, value = agent.model_old(state)

        next_state = torch.from_numpy(s1)
        next_state = next_state.float()
        _, _, _, next_value = agent.model_old(next_state)

    value = value.data.numpy()
    next_value = next_value.data.numpy()

    # Discounted Return
    gae = 0
    for t in range(NUM_STEP - 1, -1, -1):
        delta = r[t] + DISCOUNT * next_value[t] * (1 - d[t]) - value[t]
        gae = delta + DISCOUNT * LAM * (1 - d[t]) * gae
        discounted_return[t, 0] = gae + value[t]

    # For critic
    target = r + DISCOUNT * (1 - d) * next_value

    # For Actor
    adv = discounted_return - value
    # adv = (adv - adv.mean()) / (adv.std() + 1e-5)

    return [s, target, y, adv]


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCriticNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.mu = nn.Linear(256, output_size)
        self.critic = nn.Linear(256, 1)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        self.critic.weight.data.mul_(0.1)
        self.critic.bias.data.mul_(0.0)

    def forward(self, state):
        x = self.feature(state)
        mu = self.mu(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        value = self.critic(x)
        return mu, std, logstd, value


# PAAC(Parallel Advantage Actor Critic)
class ActorAgent(object):
    def __init__(self):
        self.model_old = ActorCriticNetwork(INPUT, OUTPUT)
        self.model_old.share_memory()

        self.output_size = OUTPUT
        self.input_size = INPUT

    def get_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        state = state.float()
        mu, std, logstd, value = self.model_old(state)
        m = Normal(loc=mu,scale=std)
        action = m.sample()
        return action.item()

    # after some time interval update the target model to be same with model
    def update_actor_model(self, target):
        self.model_old.load_state_dict(target.state_dict())

    @staticmethod
    def weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Linear') != -1:
            torch.nn.init.kaiming_uniform(m.weight)
            print(m)
        elif class_name.find('Conv') != -1:
            torch.nn.init.kaiming_uniform(m.weight)
            print(m)


class LearnerAgent(object):
    def __init__(self):
        self.model = ActorCriticNetwork(INPUT, OUTPUT)
        # self.model.cuda()
        self.output_size = OUTPUT
        self.input_size = INPUT
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, eps=1e-5)

    def train_model(self, s_batch, target_batch, y_batch, adv_batch, actor_agent):
        s_batch = torch.FloatTensor(s_batch)
        target_batch = torch.FloatTensor(target_batch)
        adv_batch = torch.FloatTensor(adv_batch)
        with torch.no_grad():
            mu_old, std_old, logstd_old, value_old = actor_agent.model_old(s_batch)
            m_old = Normal(loc=mu_old, scale=std_old)
            y_batch_old = torch.FloatTensor(y_batch)
            log_prob_old = m_old.log_prob(y_batch_old)

        # for multiply advantage
        mu, std, logstd, value = self.model(s_batch)
        m = Normal(loc=mu, scale=std)
        y_batch = m.sample()
        log_prob = m.log_prob(y_batch)
        entropy = m.entropy().mean()

        for i in range(EPOCH):
            minibatch = random.sample(range(len(s_batch)), BATCH_SIZE)
            ratio = torch.exp(log_prob[minibatch] - log_prob_old[minibatch])

            surr1 = ratio * adv_batch[minibatch,0].sum(0)
            surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * adv_batch[minibatch,0].sum(0)

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(value_old[minibatch], target_batch[minibatch])

            self.optimizer.zero_grad()
            loss = actor_loss + V_COEF * critic_loss - 0.0 * entropy
            loss.backward(retain_graph=True)
            self.optimizer.step()


class Environment(object):
    def __init__(self, env, idx):
        self.env = env
        self.obs = self.env.reset()
        self.next_obs = None
        self.done = False
        self.env_idx = idx
        self.step = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.recent_rlist.append(0)

    def run(self, agent):
        sample = []
        for _ in range(NUM_STEP):
            self.step += 1
            action = agent.get_action(self.obs)
            self.next_obs, reward, self.done, _ = self.env.step([action])
            self.rall += reward

            # # negative reward
            # if self.done and self.step < self.env.spec.timestep_limit:
            #     reward = 0

            sample.append([self.obs[:], action, reward, self.next_obs[:], self.done])

            self.obs = self.next_obs

            if self.done:
                self.episode += 1
                if self.env_idx == 0:
                    self.recent_rlist.append(self.rall)
                    print("[Episode {0:6d}] Reward: {1:4.2f}  Recent Reward: {2:4.2f}"
                          .format(self.episode, self.rall, np.mean(self.recent_rlist)))

                self.obs = self.env.reset()
                self.done = False
                self.step = 0
                self.rall = 0

        return make_batch(sample, agent)


def runner(env, cond, memory, actor):
    while True:
        with cond:
            sample = env.run(actor)
            memory.put(sample)

            # wait runner
            cond.wait()


def learner(cond, memory, actor_agent, learner_agent):
    while True:
        if memory.full():
            s_batch, target_batch, y_batch, adv_batch = [], [], [], []
            # while memory.qsize() != 0:
            # if you use MacOS, use under condition.
            if NUM_ENV == 1:
                batch = memory.get()
                s_batch.extend(batch[0])
                target_batch.extend(batch[1])
                y_batch.extend(batch[2])
                adv_batch.extend(batch[3])
            else:
                while not memory.empty():
                    batch = memory.get()
                    s_batch.extend(batch[0])
                    target_batch.extend(batch[1])
                    y_batch.extend(batch[2])
                    adv_batch.extend(batch[3])

            # train
            learner_agent.train_model(s_batch, target_batch, y_batch, adv_batch, actor_agent)
            actor_agent.update_actor_model(learner_agent.model)
            # resume running
            with cond:
                cond.notify_all()


def main():
    num_envs = NUM_ENV
    memory = mp.Queue(maxsize=NUM_ENV)
    cond = mp.Condition()

    # make agent and share memory
    actor_agent = ActorAgent()
    learner_agent = LearnerAgent()

    # sync model
    actor_agent.update_actor_model(learner_agent.model)

    # make envs
    envs = [Environment(gym.make(ENV_ID), i) for i in range(num_envs)]

    # Learner Process(only Learn)
    learn_proc = mp.Process(target=learner, args=(cond, memory, actor_agent, learner_agent))

    # Runner Process(just run, not learn)
    runners = []
    for idx, env in enumerate(envs):
        run_proc = mp.Process(target=runner, args=(env, cond, memory, actor_agent))
        runners.append(run_proc)
        run_proc.start()

    learn_proc.start()

    for proc in runners:
        proc.join()

    learn_proc.join()


if __name__ == '__main__':
    torch.manual_seed(23)
    ENV_ID = 'Pendulum-v0'
    env = gym.make(ENV_ID)
    # Hyper parameter
    INPUT = env.observation_space.shape[0]
    OUTPUT = env.action_space.shape[0]
    DISCOUNT = 0.99
    NUM_STEP = 2048
    NUM_ENV = 1
    LAM = 0.95
    EPOCH = 10
    BATCH_SIZE = 64
    V_COEF = 1.0
    EPSILON = 0.2
    ALPHA = 0.99
    LEARNING_RATE = 0.0003
    env.close()

    main()