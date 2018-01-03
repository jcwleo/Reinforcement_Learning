# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import gym
from collections import deque


def make_batch(A2Cagent, sample):
    sample = np.stack(sample)
    discounted_return = np.empty([NSTEP, 1])

    s = np.reshape(np.stack(sample[:, 0]), [NSTEP, A2Cagent.input_size])
    s1 = np.reshape(np.stack(sample[:, 3]), [NSTEP, A2Cagent.input_size])
    y = np.reshape(np.stack(sample[:, 1]), [NSTEP, A2Cagent.output_size])
    r = np.reshape(np.stack(sample[:, 2]), [NSTEP, 1])
    d = np.reshape(np.stack(sample[:, 4]), [NSTEP, 1])

    value = A2Cagent.sess.run(A2Cagent.v, feed_dict={A2Cagent.X: s})
    next_value = A2Cagent.sess.run(A2Cagent.v, feed_dict={A2Cagent.X: s1})

    # Discounted Return 계산
    running_add = next_value[NSTEP - 1, 0] * d[NSTEP - 1, 0]
    for t in range(4, -1, -1):
        if d[t]:
            running_add = 0
        running_add = r[t] + DISCOUNT * running_add
        discounted_return[t, 0] = running_add

    # For critic
    target = r + DISCOUNT * d * next_value

    # For Actor
    adv = discounted_return - value

    return [s, target, y, adv]


class ActorCritic:
    def __init__(self, sess, input_size, output_size):
        self.sess = sess
        self.input_size = input_size
        self.output_size = output_size

        self.build_network()

    def build_network(self):
        self.X = tf.placeholder('float', [None, self.input_size])
        self.Y = tf.placeholder('float', [None, self.output_size])
        self.adv = tf.placeholder('float')
        self.r = tf.placeholder('float')
        self.LR = tf.placeholder('float')

        # Common Weight
        w1 = tf.get_variable('w1', shape=[self.input_size, 128], initializer=tf.contrib.layers.xavier_initializer())

        # Actor Weight
        w2_a = tf.get_variable('w2_a', shape=[128, self.output_size], initializer=tf.contrib.layers.xavier_initializer())

        # Critic Weight
        w2_c = tf.get_variable('w2_c', shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())

        # Common Layer
        l1 = tf.nn.selu(tf.matmul(self.X, w1))

        # Actor Output
        self.a = tf.matmul(l1, w2_a)
        self.a_prob = tf.nn.softmax(tf.matmul(l1, w2_a))

        # Critic Output
        self.v = tf.matmul(l1, w2_c)

        # Actor loss
        self.log_lik = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.a)
        self.p_loss = tf.reduce_mean(self.log_lik * self.adv)

        # Critic loss
        self.v_loss = tf.reduce_mean(tf.square(self.v - self.r), axis=1)

        # entropy(for more exploration)
        self.entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.a_prob, logits=self.a))

        self.loss = self.p_loss + self.v_loss - self.entropy * 0.01

        self.train = tf.train.RMSPropOptimizer(learning_rate=self.LR, epsilon=EPSILON).minimize(self.loss)

    def get_action(self, state):
        state_t = np.reshape(state, [1, self.input_size])
        action_p = self.sess.run(self.a_prob, feed_dict={self.X: state_t})

        # 각 액션의 확률로 액션을 결정
        action = np.random.choice(np.arange(self.output_size), p=action_p[0])

        return action


class Runner:
    def __init__(self, idx):
        self.env = gym.make('CartPole-v1')

        self.done = False
        self.s = self.env.reset()
        self.s1 = None
        self.sample = []
        self.step = 0
        self.runner_idx = idx
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.recent_rlist.append(0)

    def run(self, A2Cagent):
        if self.done:
            self.episode += 1
            if self.runner_idx == 0:
                self.recent_rlist.append(self.rall)
                print("[Episode {0:6d}] Reward: {1:4.2f}  Recent Reward: {2:4.2f}".format(self.episode, self.rall,
                                                                                     np.mean(self.recent_rlist)))
            self.done = False
            self.rall = 0
            self.step = 0
            self.s = self.env.reset()

        self.step += 1
        action = A2Cagent.get_action(self.s)

        # action을 one_hot으로 표현
        y = np.zeros(OUTPUT)
        y[action] = 1
        s1, reward, self.done, _ = self.env.step(action)

        self.rall += reward

        # negative reward
        if self.done and self.step < self.env.spec.timestep_limit:
            reward = -100

        self.sample.append([self.s, y, reward, s1, self.done])
        self.s = s1


def main():
    with tf.Session() as sess:
        A2Cagent = ActorCritic(sess, INPUT, OUTPUT)
        A2Cagent.sess.run(tf.global_variables_initializer())

        step = 0
        runners = [Runner(i) for i in range(NENV)]

        while np.mean(runners[0].recent_rlist) <= 495:
            s_batch = []
            target_batch = []
            y_batch = []
            adv_batch = []

            learning_rate = LEARNING_RATE

            for t in range(NSTEP):
                for i in range(NENV):
                    runners[i].run(A2Cagent)

            for i in range(NENV):
                batch = make_batch(A2Cagent, runners[i].sample)

                s_batch.extend(batch[0])
                target_batch.extend(batch[1])
                y_batch.extend(batch[2])
                adv_batch.extend(batch[3])

                runners[i].sample = []

            feed_dict = {A2Cagent.X: s_batch, A2Cagent.r: target_batch, A2Cagent.Y: y_batch, A2Cagent.adv: adv_batch,
                         A2Cagent.LR: learning_rate}

            # Train Network
            A2Cagent.sess.run([A2Cagent.train], feed_dict=feed_dict)

            step += NENV * NSTEP


if __name__ == "__main__":
    env = gym.make('CartPole-v1')

    # 하이퍼 파라미터
    INPUT = env.observation_space.shape[0]
    OUTPUT = env.action_space.n
    DISCOUNT = 0.99
    NSTEP = 5
    NENV = 8
    EPSILON = 0.1
    LEARNING_RATE = 7e-4 * NENV
    env.close()
    main()
