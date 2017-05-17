# -*- coding: utf-8 -*-
# 김성훈님 ( https://github.com/hunkim/ReinforcementZeroToAll/blob/master/07_3_dqn_2015_cartpole.py )
# 김태훈님 ( https://github.com/devsisters/DQN-tensorflow )
# 코드를 참조했습니다. 감사합니다!
#
import tensorflow as tf
import gym

import numpy as np
import random as ran
import datetime
import matplotlib.pyplot as plt

from collections import deque
from skimage.transform import resize
from skimage.color import rgb2gray

plt.ion()
# DQN paper setting(frameskip = 4, repeat_action_probability = 0)
# {}Deterministic : frameskip = 4
# {}-v4 : repeat_action_probability
env = gym.make('BreakoutDeterministic-v4')

# 하이퍼 파라미터
MINIBATCH_SIZE = 32
HISTORY_SIZE = 4
TRAIN_START = 50000
FINAL_EXPLORATION = 0.1
TARGET_UPDATE = 10000
MEMORY_SIZE = 400000
EXPLORATION = 1000000
START_EXPLORATION = 1.
INPUT = env.observation_space.shape
OUTPUT = env.observation_space.n
HEIGHT = 84
WIDTH = 84
LEARNING_RATE = 0.00025
DISCOUNT = 0.99
EPSILON = 0.01
MOMENTUM = 0.95

# 트레이닝된 모델 경로
model_path = "save/Breakout.ckpt"


def cliped_error(error):
    '''후버로스를 사용하여 error 클립.

    Args:
        error(tensor): 클립을 해야할 tensor

    Returns:
        tensor: -1 ~ 1 사이로 클립된 error
    '''
    return tf.where(tf.abs(error) < 1.0, 0.5 * tf.square(error), tf.abs(error) - 0.5)


def pre_proc(X):
    '''입력데이터 전처리.

    Args:
        X(np.array): 받아온 이미지를 그레이 스케일링 후 84X84로 크기변경
            그리고 정수값으로 저장하기위해(메모리 효율 높이기 위해) 255를 곱함

    Returns:
        np.array: 변경된 이미지
    '''
    # 바로 전 frame과 비교하여 max를 취함으로써 flickering을 제거
    # x = np.maximum(X, X1)
    # 그레이 스케일링과 리사이징을 하여 데이터 크기 수정
    x = np.uint8(resize(rgb2gray(X), (HEIGHT, WIDTH), mode='reflect') * 255)
    return x


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


def get_init_state(history, s):
    '''에피소드 시작 State를 초기화.

    Args:
        history(np.array): 5개의 프레임이 저장될 array
        s(list): 초기화된 이미지

    Note:
        history[:,:,:3]에 모두 초기화된 이미지(s)를 넣어줌
    '''
    for i in range(HISTORY_SIZE):
        history[:, :, i] = pre_proc(s)


def get_game_type(count, l, no_life_game, start_live):
    '''라이프가 있는 게임인지 판별

    Args:
        count(int): 에피소드 시작 후 첫 프레임인지 확인하기 위한 arg
        l(dict): 라이프 값들이 저장되어있는 dict ex) l['ale.lives']
        no_life_game(bool): 라이프가 있는 게임일 경우, bool 값을 반환해주기 위한 arg
        start_live(int): 라이프가 있는 경우 라이프값을 초기화 하기 위한 arg

    Returns:
        list:
            no_life_game(bool): 라이프가 없는 게임이면 True, 있으면 False
            start_live(int): 라이프가 있는 게임이면 초기화된 라이프
    '''
    if count == 1:
        start_live = l['ale.lives']
        # 시작 라이프가 0일 경우, 라이프 없는 게임
        if start_live == 0:
            no_life_game = True
        else:
            no_life_game = False
    return [no_life_game, start_live]


def get_terminal(start_live, l, reward, no_life_game, ter):
    '''목숨이 줄어들거나, negative reward를 받았을 때, terminal 처리

    Args:
        start_live(int): 라이프가 있는 게임일 경우, 현재 라이프 수
        l(dict): 다음 상태에서 라이프가 줄었는지 확인하기 위한 다음 frame의 라이프 info
        no_life_game(bool): 라이프가 없는 게임일 경우, negative reward를 받으면 terminal 처리를 해주기 위한 게임 타입
        ter(bool): terminal 처리를 저장할 arg

    Returns:
        list:
            ter(bool): terminal 상태
            start_live(int): 줄어든 라이프로 업데이트된 값
    '''
    if no_life_game:
        # 목숨이 없는 게임일 경우 Terminal 처리
        if reward < 0:
            ter = True
    else:
        # 목숨 있는 게임일 경우 Terminal 처리
        if start_live > l['ale.lives']:
            ter = True
            start_live = l['ale.lives']

    return [ter, start_live]


def train_minibatch(mainDQN, targetDQN, minibatch):
    '''미니배치로 가져온 sample데이터로 메인네트워크 학습

    Args:
        mainDQN(object): 메인 네트워크
        targetDQN(object): 타겟 네트워크
        minibatch: replay_memory에서 MINIBATCH 개수만큼 랜덤 sampling 해온 값

    Note:
        replay_memory에서 꺼내온 값으로 메인 네트워크를 학습
    '''
    s_stack = []
    a_stack = []
    r_stack = []
    s1_stack = []
    d_stack = []

    for s_r, a_r, r_r, d_r in minibatch:
        s_stack.append(s_r[:, :, :4])
        a_stack.append(a_r)
        r_stack.append(r_r)
        s1_stack.append(s_r[:, :, 1:])
        d_stack.append(d_r)

    # True, False 값을 1과 0으로 변환
    d_stack = np.array(d_stack) + 0

    Q1 = targetDQN.get_q(np.array(s1_stack))

    y = r_stack + (1 - d_stack) * DISCOUNT * np.max(Q1, axis=1)

    # 업데이트 된 Q값으로 main네트워크를 학습
    mainDQN.sess.run(mainDQN.train, feed_dict={mainDQN.X: np.float32(np.array(s_stack) / 255.), mainDQN.Y: y,
                                               mainDQN.a: a_stack})


# 데이터 플롯
def plot_data(epoch, epoch_score, average_reward, epoch_Q, average_Q, mainDQN):
    plt.clf()
    epoch_score.append(np.mean(average_reward))
    epoch_Q.append(np.mean(average_Q))

    plt.subplot(211)
    plt.axis([0, epoch, 0, np.max(epoch_Q) * 6 / 5])
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Action Value(Q)')
    plt.plot(epoch_Q)

    plt.subplot(212)
    plt.axis([0, epoch, 0, np.max(epoch_score) * 6 / 5])
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Reward per Episode')
    plt.plot(epoch_score, "r")

    plt.pause(0.05)
    plt.savefig("graph/{} epoch".format(epoch - 1))

    save_path = mainDQN.saver.save(mainDQN.sess, model_path, global_step=(epoch - 1))
    print("Model(epoch :", epoch, ") saved in file: ", save_path, " Now time : ", datetime.datetime.now())


# DQN
class DQNAgent:
    def __init__(self, sess, HEIGHT, WIDTH, HISTORY_SIZE, OUTPUT, NAME='main'):
        self.sess = sess
        self.height = HEIGHT
        self.width = WIDTH
        self.history_size = HISTORY_SIZE
        self.output = OUTPUT
        self.name = NAME

        self.build_network()

    def build_network(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder('float', [None, self.height, self.width, self.history_size])
            self.Y = tf.placeholder('float', [None])
            self.a = tf.placeholder('int64', [None])

            f1 = tf.get_variable("f1", shape=[8, 8, 4, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            f2 = tf.get_variable("f2", shape=[4, 4, 32, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            f3 = tf.get_variable("f3", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            w1 = tf.get_variable("w1", shape=[7 * 7 * 64, 512], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable("w2", shape=[512, OUTPUT], initializer=tf.contrib.layers.xavier_initializer())

            c1 = tf.nn.relu(tf.nn.conv2d(self.X, f1, strides=[1, 4, 4, 1], padding="VALID"))
            c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1], padding="VALID"))
            c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1, 1, 1, 1], padding='VALID'))

            l1 = tf.reshape(c3, [-1, w1.get_shape().as_list()[0]])
            l2 = tf.nn.relu(tf.matmul(l1, w1))

            self.Q_pre = tf.matmul(l2, w2)

        a_one_hot = tf.one_hot(self.a, self.output, 1.0, 0.0)
        q_val = tf.reduce_sum(tf.multiply(self.Q_pre, a_one_hot), reduction_indices=1)

        # error를 -1~1 사이로 클립
        error = cliped_error(self.Y - q_val)

        self.loss = tf.reduce_mean(error)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=EPSILON)
        self.train = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=None)

    def get_q(self, history):
        return self.sess.run(self.Q_pre, feed_dict={self.X: np.reshape(np.float32(history / 255.),
                                                                       [-1, 84, 84, 4])})

    def get_action(self, q, e):
        if e > np.random.rand(1):
            action = np.random.randint(self.output)
        else:
            action = np.argmax(q)
        return action


def main():
    with tf.Session() as sess:
        mainDQN = DQNAgent(sess, HEIGHT, WIDTH, HISTORY_SIZE, OUTPUT, NAME='main')
        targetDQN = DQNAgent(sess, HEIGHT, WIDTH, HISTORY_SIZE, OUTPUT, NAME='target')

        mainDQN.saver.restore(sess, model_path)
        recent_rlist = deque(maxlen=100)
        e = 1.
        episode, epoch, frame = 0, 0, 0

        average_Q, average_reward = deque(), deque()

        # Train agent during 200 epoch
        while epoch <= 200:
            episode += 1

            history = np.zeros([84, 84, 5], dtype=np.uint8)
            rall, count = 0, 0
            d = False
            s = env.reset()

            get_init_state(history, s)
            while not d:
                # env.render()

                frame += 1
                count += 1

                # e-greedy
                if e > FINAL_EXPLORATION and frame > TRAIN_START:
                    e -= (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

                # 히스토리의 0~4까지 부분으로 Q값 예측
                Q = mainDQN.get_q(history[:, :, :4])
                average_Q.append(np.max(Q))

                # 액션 선택
                action = mainDQN.get_action(Q, e)

                # s1 : next frame / r : reward / d : done(terminal) / l : info(lives)
                s1, r, d, l = env.step(action)

                # 새로운 프레임을 히스토리 마지막에 넣어줌
                history[:, :, 4] = pre_proc(s1)

                history[:, :, :4] = history[:, :, 1:]

                rall += r

            recent_rlist.append(rall)

            average_reward.append(rall)

            print("Episode:{0:6d} | Frames:{1:9d} | Steps:{2:5d} | Reward:{3:3.0f} | e-greedy:{4:.5f} | "
                  "Avg_Max_Q:{5:2.5f} | Recent reward:{6:.5f}  ".format(episode, frame, count, rall, e,
                                                                        np.mean(average_Q),
                                                                        np.mean(recent_rlist)))


if __name__ == "__main__":
    main()





