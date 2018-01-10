# -*- coding: utf-8 -*-
import math
import random as ran

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

plt.ion()
# DQN paper setting(frameskip = 4, repeat_action_probability = 0)
# {}Deterministic : frameskip = 4
# {}-v4 : repeat_action_probability
env = gym.make('CartPole-v1')

# 하이퍼 파라미터
MINIBATCH_SIZE = 32
TRAIN_START = 1000
FINAL_EXPLORATION = 0.01
TARGET_UPDATE = 1000
MEMORY_SIZE = 50000
EXPLORATION = 20000
START_EXPLORATION = 1.
INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n
LEARNING_RATE = 0.001
DISCOUNT = 0.99
EPSILON = 0.01
MOMENTUM = 0.95
VMIN = -50
VMAX = 100
CATEGORY = 51

model_path = "save/CartPole_C51.ckpt"


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    '''타겟네트워크에 메인네트워크의 Weight값을 복사.

    Args:
        dest_scope_name="target"(DQN): 'target'이라는 이름을 가진 객체를 가져옴
        src_scope_name="main"(DQN): 'main'이라는 이름을 가진 객체를 가져옴

    Returns:
        list: main의 trainable한 값들이 target의 값으로 복사된 값
    '''
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def train_minibatch(mainC51, targetC51, minibatch):
    '''미니배치로 가져온 sample데이터로 메인네트워크 학습

    Args:
        mainC51(object): 메인 네트워크
        targetC51(object): 타겟 네트워크
        minibatch: replay_memory에서 MINIBATCH 개수만큼 랜덤 sampling 해온 값

    Note:
        replay_memory에서 꺼내온 값으로 메인 네트워크를 학습
    '''
    s_stack = []
    a_stack = []
    r_stack = []
    s1_stack = []
    d_stack = []
    m_prob = [np.zeros((len(minibatch), mainC51.category_size)) for i in range(OUTPUT)]

    for s_r, a_r, r_r, d_r, s1_r in minibatch:
        s_stack.append(s_r)
        a_stack.append(a_r)
        r_stack.append(r_r)
        s1_stack.append(s1_r)
        d_stack.append(d_r)

    # Category Algorighm
    target_sum_q = targetC51.sess.run(targetC51.dist_Q, feed_dict={targetC51.X: np.vstack(s1_stack)})

    # Get optimal action
    sum_q = mainC51.optimal_action(s1_stack)
    sum_q = sum_q.reshape([len(s_stack), OUTPUT], order='F')
    optimal_action = np.argmax(sum_q, axis=1)

    for i in range(len(s_stack)):
        if d_stack[i]:
            Tz = min(VMAX, max(VMIN, r_stack[i]))
            bj = (Tz - VMIN) / mainC51.delta_z
            m_l, m_u = math.floor(bj), math.ceil(bj)
            m_prob[a_stack[i]][i][int(m_l)] += (m_u - bj)
            m_prob[a_stack[i]][i][int(m_u)] += (bj - m_l)
        else:
            for j in range(mainC51.category_size):
                Tz = min(VMAX, max(VMIN, r_stack[i] + DISCOUNT * mainC51.z[j]))
                bj = (Tz - VMIN) / mainC51.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[a_stack[i]][i][int(m_l)] += (m_u - bj) * target_sum_q[optimal_action[i]][i][j]
                m_prob[a_stack[i]][i][int(m_u)] += (bj - m_l) * target_sum_q[optimal_action[i]][i][j]

    mainC51.sess.run(mainC51.train, feed_dict={mainC51.X: np.vstack(s_stack), mainC51.Y: m_prob})


class C51Agent:
    def __init__(self, sess, INPUT, OUTPUT, VMAX, VMIN, CATEGORY, NAME='main'):
        self.sess = sess

        self.input_size = INPUT
        self.output_size = OUTPUT
        self.category_size = CATEGORY
        self.delta_z = (VMAX - VMIN) / float(self.category_size - 1)
        self.z = [VMIN + i * self.delta_z for i in range(self.category_size)]
        self.name = NAME

        self.build_network()

    def build_network(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder('float', [None, self.input_size])
            self.Y = tf.placeholder('float', [2, None, self.category_size])

            self.dist_Q = []

            w1 = tf.get_variable("w1", shape=[self.input_size, 256], initializer=tf.contrib.layers.xavier_initializer())

            # Output weight
            for i in range(self.output_size):
                exec(
                    'w2_%s = tf.get_variable("w2_%s", shape=[256, self.category_size], initializer=tf.contrib.layers.xavier_initializer())' % (
                    i, i))

            l1 = tf.nn.relu(tf.matmul(self.X, w1))
            for i in range(self.output_size):
                exec('self.dist_Q.append(tf.matmul(l1, w2_%s))' % i)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.dist_Q))
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=EPSILON)
        self.train = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=None)

    def get_action(self, state, e):
        if e > np.random.rand(1):
            action = np.random.randint(self.output_size)
        else:
            sum_q = self.optimal_action(state)
            action = np.argmax(sum_q)
        return action

    def optimal_action(self, state):
        state = np.vstack(state)
        state = state.reshape([-1, self.input_size])
        z = self.sess.run(self.dist_Q, feed_dict={self.X: state})
        z_stack = np.vstack(z)
        sum_q = np.sum(np.multiply(z_stack, np.array(self.z)), axis=1)
        return sum_q


def main():
    with tf.Session() as sess:
        mainC51 = C51Agent(sess, INPUT, OUTPUT, VMAX, VMIN, CATEGORY, NAME='main')
        targetC51 = C51Agent(sess, INPUT, OUTPUT, VMAX, VMIN, CATEGORY, NAME='target')

        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        recent_rlist = deque(maxlen=100)
        recent_rlist.append(0)
        e = 1.
        episode, epoch, frame = 0, 0, 0

        replay_memory = deque(maxlen=MEMORY_SIZE)

        # Train agent
        while np.mean(recent_rlist) <= 495:
            episode += 1

            rall, count = 0, 0
            d = False
            s = env.reset()

            while not d:
                frame += 1
                count += 1

                # e-greedy
                if e > FINAL_EXPLORATION and frame > TRAIN_START:
                    e -= (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

                # 액션 선택
                action = mainC51.get_action(s, e)

                # s1 : next frame / r : reward / d : done(terminal) / l : info(lives)
                s1, r, d, l = env.step(action)
                if d and count < env.spec.timestep_limit:
                    reward = -100
                else:
                    reward = r

                replay_memory.append((s, action, reward, d, s1))
                s = s1

                rall += r

                if frame > TRAIN_START:
                    minibatch = ran.sample(replay_memory, MINIBATCH_SIZE)
                    train_minibatch(mainC51, targetC51, minibatch)

                    if frame % TARGET_UPDATE == 0:
                        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                                    src_scope_name="main")
                        sess.run(copy_ops)

            recent_rlist.append(rall)

            print("Episode:{0:6d} | Frames:{1:9d} | Steps:{2:5d} | Reward:{3:3.0f} | e-greedy:{4:.5f} | "
                  "Recent reward:{5:.5f}  ".format(episode, frame, count, rall, e,
                                                   np.mean(recent_rlist)))


if __name__ == "__main__":
    main()
