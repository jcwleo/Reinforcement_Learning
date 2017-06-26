# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import gym
from collections import deque

env = gym.make('CartPole-v1')

# 하이퍼 파라미터
LEARNING_RATE = 0.001
INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n
DISCOUNT = 0.99
print(env.spec.timestep_limit)

def train_onestep(A2Cagent, s, y, r, s1, d):
    s = np.reshape(s, [1, A2Cagent.input_size])
    s1 = np.reshape(s1, [1, A2Cagent.input_size])
    y = np.reshape(y, [1, A2Cagent.output_size])

    value = A2Cagent.sess.run(A2Cagent.v, feed_dict={A2Cagent.X: s})
    if d:
        target = r
        adv = r - value
    else:
        next_value = A2Cagent.sess.run(A2Cagent.v, feed_dict={A2Cagent.X : s1})
        target = r + DISCOUNT * next_value
        adv = r + DISCOUNT * next_value - value

    A2Cagent.sess.run([A2Cagent.train], feed_dict={A2Cagent.X: s, A2Cagent.r: target, A2Cagent.Y: y, A2Cagent.adv: adv})

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

        # Actor Weight
        w1_a = tf.get_variable('w1', shape=[self.input_size, 128], initializer=tf.contrib.layers.xavier_initializer())
        w2_a = tf.get_variable('w2', shape=[128, self.output_size], initializer=tf.contrib.layers.xavier_initializer())

        # Critic Weight
        w1_c = tf.get_variable('w1_c', shape=[self.input_size, 128], initializer=tf.contrib.layers.xavier_initializer())
        w2_c = tf.get_variable('w2_c', shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())

        # Actor Critic Network
        l1_a = tf.nn.relu(tf.matmul(self.X, w1_a))
        l1_c = tf.nn.relu(tf.matmul(self.X, w1_c))
        self.a_prob = tf.nn.softmax(tf.matmul(l1_a, w2_a))
        self.v = tf.matmul(l1_c, w2_c)

        # Policy loss
        self.log_p = self.Y * tf.log(tf.clip_by_value(self.a_prob,1e-10,1.))
        self.log_lik = self.log_p * self.adv
        self.p_loss = -tf.reduce_mean(tf.reduce_sum(self.log_lik, axis=1))

        # Value loss
        self.v_loss = tf.reduce_mean(tf.square(self.v - self.r), axis=1)

        # entropy(for more exploration)
        self.entropy = -tf.reduce_mean(
            tf.reduce_sum(self.a_prob * tf.log(tf.clip_by_value(self.a_prob, 1e-10, 1.)), axis=1))

        self.loss = self.p_loss + self.v_loss - self.entropy * 0.01
        self.train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    def get_action(self, state):
        state_t = np.reshape(state, [1, self.input_size])
        action_p = self.sess.run(self.a_prob, feed_dict={self.X: state_t})

        # 각 액션의 확률로 액션을 결정
        action = np.random.choice(np.arange(self.output_size), p=action_p[0])

        return action


def main():
    with tf.Session() as sess:
        A2Cagent = ActorCritic(sess, INPUT, OUTPUT)

        A2Cagent.sess.run(tf.global_variables_initializer())
        episode = 0
        recent_rlist = deque(maxlen=100)
        recent_rlist.append(0)

        # 최근 100개의 점수가 195점 넘을 때까지 학습
        while np.mean(recent_rlist) <= 195:
            episode += 1

            rall = 0
            count = 0
            s = env.reset()
            done = False
            i = 1
            while not done:
                count += 1
                # 액션 선택
                action = A2Cagent.get_action(s)

                # action을 one_hot으로 표현
                y = np.zeros(OUTPUT)
                y[action] = 1

                s1, reward, done, _ = env.step(action)
                rall += reward

                # negative reward
                if done and count < env.spec.timestep_limit:
                    reward = -100

                train_onestep(A2Cagent, s, y, reward, s1, done)

                s = s1

            recent_rlist.append(rall)

            print("[Episode {0:6d}] Reward: {1:4f} Recent Reward: {2:4f}".format(episode, rall, np.mean(recent_rlist)))

if __name__ == "__main__":
    main()





