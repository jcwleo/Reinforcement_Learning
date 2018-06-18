import gym
import numpy as np

import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import torch.nn as nn
import torch

from collections import deque

from torch.distributions.categorical import Categorical


def make_batch(sample, agent):
    sample = np.stack(sample)
    discounted_return = np.empty([NUM_STEP, 1])

    s = np.reshape(np.stack(sample[:, 0]), [NUM_STEP, agent.input_size])
    s1 = np.reshape(np.stack(sample[:, 3]), [NUM_STEP, agent.input_size])
    y = sample[:, 1]
    r = np.reshape(np.stack(sample[:, 2]), [NUM_STEP, 1])
    d = np.reshape(np.stack(sample[:, 4]), [NUM_STEP, 1])

    state = torch.from_numpy(s)
    state = state.float()
    _, value = agent.model(state)

    next_state = torch.from_numpy(s1)
    next_state = next_state.float()
    _, next_value = agent.model(next_state)

    value = value.data.numpy()
    next_value = next_value.data.numpy()

    # Discounted Return
    running_add = next_value[NUM_STEP - 1, 0] * d[NUM_STEP - 1, 0]
    for t in range(NUM_STEP - 1, -1, -1):
        if d[t]:
            running_add = 0
        running_add = r[t] + DISCOUNT * running_add
        discounted_return[t, 0] = running_add

    # For critic
    target = r + DISCOUNT * d * next_value

    # For Actor
    adv = discounted_return - value

    return [s, target, y, adv]


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCriticNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, output_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        x = self.feature(state)
        policy = F.softmax(self.actor(x))
        value = self.critic(x)
        return policy, value


# PAAC(Parallel Advantage Actor Critic)
class ActorAgent(object):
    def __init__(self):
        self.model = ActorCriticNetwork(INPUT, OUTPUT)

        self.model.share_memory()

        self.output_size = OUTPUT
        self.input_size = INPUT

    def get_action(self, state):
        state = torch.from_numpy(state)
        state = state.float()
        policy, value = self.model(state)
        m = Categorical(policy)
        action = m.sample()
        return action.item()

    # after some time interval update the target model to be same with model
    def update_actor_model(self, target):
        self.model.load_state_dict(target.state_dict())


class LearnerAgent(object):
    def __init__(self):
        self.model = ActorCriticNetwork(INPUT, OUTPUT)
        # self.model.cuda()
        self.output_size = OUTPUT
        self.input_size = INPUT
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def train_model(self, s_batch, target_batch, y_batch, adv_batch):
        s_batch = torch.FloatTensor(s_batch)
        target_batch = torch.FloatTensor(target_batch)
        y_batch = torch.LongTensor(y_batch)
        adv_batch = torch.FloatTensor(adv_batch)

        # for multiply advantage
        policy, value = self.model(s_batch)
        m = Categorical(policy)

        # mse = nn.SmoothL1Loss()
        mse = nn.MSELoss()

        # Actor loss
        actor_loss = -m.log_prob(y_batch) * adv_batch.sum(1)

        # Entropy(for more exploration)
        entropy = m.entropy()
        # Critic loss
        critic_loss = mse(value, target_batch)

        # Total loss
        loss = actor_loss.mean() + 0.5 * critic_loss - 0.01 * entropy.mean()
        self.optimizer.zero_grad()
        loss.backward()

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
            self.next_obs, reward, self.done, _ = self.env.step(action)
            self.rall += reward

            # negative reward
            if self.done and self.step < self.env.spec.timestep_limit:
                reward = -100

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
            while not memory.empty():
                batch = memory.get()

                s_batch.extend(batch[0])
                target_batch.extend(batch[1])
                y_batch.extend(batch[2])
                adv_batch.extend(batch[3])

            # train
            learner_agent.train_model(s_batch, target_batch, y_batch, adv_batch)
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
    envs = [Environment(gym.make('CartPole-v1'), i) for i in range(num_envs)]

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
    env = gym.make('CartPole-v1')
    # Hyper parameter
    INPUT = env.observation_space.shape[0]
    OUTPUT = env.action_space.n
    DISCOUNT = 0.99
    NUM_STEP = 5
    NUM_ENV = 8
    EPSILON = 1e-5
    ALPHA = 0.99
    LEARNING_RATE = 0.001
    env.close()

    main()
