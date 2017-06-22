# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import gym
from collections import deque

env = gym.make('CartPole-v0')

# 하이퍼 파라미터
LEARNING_RATE = 0.005
INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n
DISCOUNT = 0.99


def discount_rewards(r):
    '''Discounted reward를 구하기 위한 함수

    Args:
         r(np.array): reward 값이 저장된 array

    Returns:
        discounted_r(np.array): Discounted 된 reward가 저장된 array
    '''
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * DISCOUNT + r[t]
        discounted_r[t] = running_add

    return discounted_r


def train_episodic(A2Cagent, x, y, r):
    '''에피소드당 학습을 하기위한 함수

    Args:
        A2Cagent(ActorCritic): 학습될 네트워크
        x(np.array): State가 저장되어있는 array
        y(np.array): Action(one_hot)이 저장되어있는 array
        r(np.array) : Discounted reward가 저장되어있는 array

    Returns:
        l(float): 네트워크에 의한 loss
    '''
    l, _ = A2Cagent.sess.run([A2Cagent.loss, A2Cagent.train], feed_dict={A2Cagent.X: x, A2Cagent.Y: y, A2Cagent.r: r})
    return l


def play_cartpole(A2Cagent):
    '''학습된 네트워크로 Play하기 위한 함수

    Args:
         A2Cagent(ActorCritic): 학습된 네트워크
    '''
    print("Play Cartpole!")
    episode = 0
    while True:
        s = env.reset()
        done = False
        rall = 0
        episode += 1
        while not done:
            env.render()
            action_p = A2Cagent.get_action(s)
            s1, reward, done, _ = env.step(action_p)
            s = s1
            rall += reward
        print("[Episode {0:6f}] Reward: {1:4f} ".format(episode, rall))


class ActorCritic:
    def __init__(self, sess, input_size, output_size):
        self.sess = sess
        self.input_size = input_size
        self.output_size = output_size

        self.build_network()

    def build_network(self):

        self.X = tf.placeholder('float', [None, self.input_size])
        self.Y = tf.placeholder('float', [None, self.output_size])

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

        # A_t = R_t - V(S_t)
        self.adv = self.r - self.v

        # Policy loss
        self.log_p = self.Y * tf.log(tf.clip_by_value(self.a_prob,1e-10,1.))
        self.log_lik = self.log_p * tf.stop_gradient(self.adv)
        self.p_loss = -tf.reduce_mean(tf.reduce_sum(self.log_lik, axis=1))

        # entropy(for more exploration)
        self.entropy = -tf.reduce_mean(tf.reduce_sum(self.a_prob * tf.log(tf.clip_by_value(self.a_prob,1e-10,1.)), axis=1))

        # Value loss
        self.v_loss = tf.reduce_mean(tf.square(self.v - self.r), axis=1)

        # Total loss
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
            episode_memory = deque()
            rall = 0
            s = env.reset()
            done = False

            while not done:
                # 액션 선택
                action = A2Cagent.get_action(s)

                # action을 one_hot으로 표현
                y = np.zeros(OUTPUT)
                y[action] = 1

                s1, reward, done, _ = env.step(action)
                rall += reward

                # 에피소드 메모리에 저장
                episode_memory.append([s, y, reward])
                s = s1

                # 에피소드가 끝났을때 학습
                if done:
                    episode_memory = np.array(episode_memory)

                    discounted_rewards = discount_rewards(np.vstack(episode_memory[:, 2]))

                    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std())

                    train_episodic(A2Cagent, np.vstack(episode_memory[:, 0]), np.vstack(episode_memory[:, 1]),
                                       discounted_rewards)

                    recent_rlist.append(rall)

            print("[Episode {0:6d}] Reward: {1:4f} Recent Reward: {2:4f}".format(episode, rall, np.mean(recent_rlist)))

        play_cartpole(A2Cagent)


if __name__ == "__main__":
    main()





