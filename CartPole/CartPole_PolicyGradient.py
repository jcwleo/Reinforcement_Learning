# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import gym
from collections import deque

env = gym.make('CartPole-v0')

LEARNING_RATE = 0.005
INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n
DISCOUNT = 0.99

def discount_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * DISCOUNT + r[t]
        discounted_r[t] = running_add

    return discounted_r

def train_episode(PGagent, x, y, adv):
    l,_ = PGagent.sess.run([PGagent.loss, PGagent.train], feed_dict={PGagent.X: x, PGagent.Y: y, PGagent.adv : adv})
    return l

def play_cartpole(PGagent):
    print("Play Cartpole!")
    episode = 0
    while True:
        s = env.reset()
        done = False
        rall = 0
        episode += 1
        while not done:
            env.render()
            action = PGagent.get_action(s)
            s1, reward, done, _ = env.step(action)
            s = s1
            rall += reward
        print("[Episode {0:6f}] Reward: {1:4f} ".format(episode, rall))

class PolicyGradient:
    def __init__(self, sess, input_size, output_size):
        self.sess = sess
        self.input_size = input_size
        self.output_size = output_size

        self.build_network()

    def build_network(self):
        self.X = tf.placeholder('float',[None, self.input_size])
        self.Y = tf.placeholder('float', [None, self.output_size])
        self.adv = tf.placeholder('float')

        w1 = tf.get_variable('w1', shape=[self.input_size, 128], initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable('w2', shape=[128, self.output_size], initializer=tf.contrib.layers.xavier_initializer())

        l1 = tf.nn.relu(tf.matmul(self.X, w1))
        self.a_pre = tf.nn.softmax(tf.matmul(l1,w2))

        self.log_p = self.Y * tf.log(self.a_pre)
        self.log_adv = self.log_p * self.adv
        self.loss = tf.reduce_mean(tf.reduce_sum(-self.log_adv, axis=1))
        self.train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    def get_action(self, state):
        state_t = np.reshape(state, [1, self.input_size])
        action_p = self.sess.run(self.a_pre, feed_dict={self.X : state_t})
        action = np.random.choice(np.arange(self.output_size), p=action_p[0])

        return action

def main():
    with tf.Session() as sess:
        PGagent = PolicyGradient(sess, INPUT, OUTPUT)

        sess.run(tf.global_variables_initializer())
        episode = 0
        recent_rlist = deque(maxlen=100)
        recent_rlist.append(0)

        while np.mean(recent_rlist) <= 195:
            episode += 1
            episode_memory = deque()
            rall = 0
            s = env.reset()
            done = False

            while not done:
                action = PGagent.get_action(s)
                y = np.zeros(OUTPUT)
                y[action] = 1

                s1, reward, done, _ = env.step(action)
                rall += reward

                episode_memory.append([s, y, reward])
                s = s1
                if done:
                    episode_memory = np.array(episode_memory)

                    discounted_rewards = discount_rewards(np.vstack(episode_memory[:,2]))

                    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() +
                                                                                             1e-7)

                    l = train_episode(PGagent, np.vstack(episode_memory[:,0]), np.vstack(episode_memory[:,1]),
                                      discounted_rewards)

                    recent_rlist.append(rall)

            print("[Episode {0:6f}] Reward: {1:4f} Loss: {2:5.5f} Recent Reward: {3:4f}".format(episode, rall, l,
                                                                                                np.mean(recent_rlist)))

        play_cartpole(PGagent)

if __name__ == "__main__":
    main()





