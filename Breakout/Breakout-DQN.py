# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
import copy
import numpy as np
import random as ran
import datetime
import matplotlib.pyplot as plt

from collections import deque
from skimage.transform import resize
from skimage.color import rgb2gray

plt.ion()
env = gym.make('BreakoutDeterministic-v3')

DDQN = False

# 꺼내서 사용할 리플레이 갯수
MINIBATCH = 32
# 리플레이를 저장할 리스트
REPLAY_MEMORY = deque()

HISTORY_STEP =4
FRAMESKIP = 4
TRAIN_INTERVAL = 4
NO_STEP = 30
TRAIN_START = 50000
if DDQN:
    FINAL_EXPLORATION = 0.01
    TARGET_UPDATE = 30000
else:
    FINAL_EXPLORATION = 0.1
    TARGET_UPDATE = 10000


MEMORY_SIZE = 200000
EXPLORATION = 1000000
START_EXPLORATION = 1.


INPUT = env.observation_space.shape
OUTPUT = 3
HEIGHT =84
WIDTH = 84

# 하이퍼파라미터
LEARNING_RATE = 0.00025

DISCOUNT = 0.99
e = 1.
frame = 0
model_path = "save/Breakout.ckpt"
def cliped_error(x):
    return tf.where(tf.abs(x) < 1.0 , 0.5 * tf.square(x), tf.abs(x)-0.5)

# input data 전처리


def pre_proc(X):
    # 바로 전 frame과 비교하여 max를 취함으로써 flickering을 제거
    # x = np.maximum(X, X1)
    # 그레이 스케일링과 리사이징을 하여 데이터 크기 수정
    x = np.uint8(resize(rgb2gray(X), (84,84))*255)
    return x

# DQN 모델
def model(input1, f1, f2, f3, w1, w2):
    c1 = tf.nn.relu(tf.nn.conv2d(input1, f1, strides=[1, 4, 4, 1],data_format="NHWC", padding = "VALID"))
    c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1],data_format="NHWC", padding="VALID"))
    c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1,1,1,1],data_format="NHWC", padding="VALID"))

    l1 = tf.reshape(c3, [-1, w1.get_shape().as_list()[0]])
    l2 = tf.nn.relu(tf.matmul(l1, w1))

    pyx = tf.matmul(l2, w2)
    return pyx


X = tf.placeholder("float", [None, 84, 84, 4])

# 메인 네트워크 Variable
f1 = tf.get_variable("f1", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
f2 = tf.get_variable("f2", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
f3 = tf.get_variable("f3", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

w1 = tf.get_variable("w1", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
w2 = tf.get_variable("w2", shape=[512, OUTPUT], initializer=tf.contrib.layers.xavier_initializer())

py_x = model(X, f1, f2, f3 , w1, w2)

# 타겟 네트워크 Variable
f1_r = tf.get_variable("f1_r", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
f2_r = tf.get_variable("f2_r", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
f3_r = tf.get_variable("f3_r", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

w1_r = tf.get_variable("w1_r", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
w2_r = tf.get_variable("w2_r", shape=[512, OUTPUT], initializer=tf.contrib.layers.xavier_initializer())

py_x_r = model(X, f1_r, f2_r,f3_r, w1_r, w2_r)

# 총 Reward를 저장해놓을 리스트
rlist=[0]
recent_rlist=[0]

episode = 0
epoch = 0
epoch_score = deque()
epoch_Q = deque()
epoch_on = False
average_Q = deque()
average_reward = deque()
no_life_game = False

# Loss function 정의
a= tf.placeholder(tf.int64, [None])
y = tf.placeholder(tf.float32, [None])
a_one_hot = tf.one_hot(a, OUTPUT, 1.0, 0.0)
q_value = tf.reduce_sum(tf.multiply(py_x, a_one_hot), reduction_indices=1)
error = tf.abs(y - q_value)

quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
linear_part = error - quadratic_part
loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE,momentum=0.95,epsilon= 0.01)
train = optimizer.minimize(loss)

saver = tf.train.Saver(max_to_keep=None)

# 세션 정의
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())
    sess.run(w1_r.assign(w1))
    sess.run(w2_r.assign(w2))
    sess.run(f1_r.assign(f1))
    sess.run(f2_r.assign(f2))
    sess.run(f3_r.assign(f3))

    # 에피소드 시작
    while np.mean(recent_rlist) < 500 :
        episode += 1

        # 가장 최근의 100개 episode의 total reward
        if len(recent_rlist) > 100:
            del recent_rlist[0]

        history = np.zeros((84, 84, 5), dtype=np.uint8)
        rall = 0
        d = False
        ter = False
        count = 0
        s = env.reset()
        avg_max_Q = 0
        avg_loss = 0

        # 에피소드 시작할때 최대 30만큼 동안 아무 행동 하지않음
        # for _ in range(ran.randint(1, NO_STEP)):
        #     s1, _, _, _ = env.step(0)

        # state의 초기화
        for i in range(HISTORY_STEP):
            history[:, :, i] = pre_proc(s)

        # 에피소드가 끝나기 전까지 반복
        while not d :
            # env.render()
            # 최근 4개의 프레임을 현재 프레임으로 바꿔줌

            frame +=1
            count+=1

            if e > FINAL_EXPLORATION and frame > TRAIN_START:
                e -= (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

            # 현재 state로 Q값을 계산
            Q = sess.run(py_x, feed_dict = {X : np.reshape(np.float32(history/255.), (1, 84, 84, 5))[:, :, :, 0:4]})
            average_Q.append(np.max(Q))
            avg_max_Q += np.max(Q)

            if e > np.random.rand(1):
                action = np.random.randint(OUTPUT)
            else:
                action = np.argmax(Q)

            if action == 0:
                real_a = 1
            elif action == 1:
                real_a = 4
            else:
                real_a = 5


            # 결정된 action으로 Environment에 입력
            s1, r, d, l = env.step(real_a)
            ter = d
            reward= np.clip(r, -1,1)


            # next state를 history에 저장
            history[:,:, 4] = pre_proc(s1)

            # 저장된 state를 Experience Replay memory에 저장
            REPLAY_MEMORY.append((np.copy(history[:,:,:]), action ,reward, ter))
            history[:,:,:4] = history[:,:,1:]

            # 저장된 Frame이 1백만개 이상 넘어가면 맨 앞 Replay부터 삭제
            if len(REPLAY_MEMORY) > MEMORY_SIZE:
                REPLAY_MEMORY.popleft()
            # 총 reward 합
            rall += r

            # 5만 frame 이상부터 4개의 Frame마다 학습
            if frame > TRAIN_START :
                s_stack = deque()
                a_stack = deque()
                r_stack = deque()
                s1_stack = deque()
                d_stack = deque()
                y_stack = deque()

                sample = ran.sample(REPLAY_MEMORY, MINIBATCH)

                for s_r, a_r, r_r, d_r in sample:
                    s_stack.append(s_r[:,:,:4])
                    a_stack.append(a_r)
                    r_stack.append(r_r)
                    s1_stack.append(s_r[:,:,1:])
                    d_stack.append(d_r)

                d_stack = np.array(d_stack) + 0

                Q1 = sess.run(py_x_r, feed_dict={X: np.float32(np.array(s1_stack) / 255.)})

                y_stack = r_stack + (1 - d_stack) * DISCOUNT * np.max(Q1, axis=1)

                # 업데이트 된 Q값으로 main네트워크를 학습
                sess.run(train, feed_dict={X: np.float32(np.array(s_stack) / 255.), y: y_stack, a: a_stack})

                # 3만개의 Frame마다 타겟 네트워크 업데이트
                if frame % TARGET_UPDATE == 0 :
                    sess.run(w1_r.assign(w1))
                    sess.run(w2_r.assign(w2))
                    sess.run(f1_r.assign(f1))
                    sess.run(f2_r.assign(f2))
                    sess.run(f3_r.assign(f3))

            # epoch(50000 Trained frame) 마다 plot
            if (frame - TRAIN_START) % 50000 == 0:
                epoch_on = True

        if epoch_on:
            plt.clf()
            epoch += 1
            epoch_score.append(np.mean(average_reward))
            epoch_Q.append(np.mean(average_Q))

            plt.subplot(211)
            plt.axis([0, epoch, 0, np.max(epoch_Q)*6/5])
            plt.xlabel('Training Epochs')
            plt.ylabel('Average Action Value(Q)')
            plt.plot(epoch_Q)

            plt.subplot(212)
            plt.axis([0, epoch , 0, np.max(epoch_score)*6/5])
            plt.xlabel('Training Epochs')
            plt.ylabel('Average Reward per Episode')
            plt.plot(epoch_score, "r")

            epoch_on = False
            average_reward = deque()
            average_Q = deque()
            plt.pause(0.05)
            plt.savefig("graph/{} epoch".format(epoch-1))

            save_path = saver.save(sess, model_path, global_step=(epoch-1))
            print("Model(episode :",episode, ") saved in file: ", save_path , " Now time : " ,datetime.datetime.now())



        # 총 reward의 합을 list에 저장
        recent_rlist.append(rall)
        rlist.append(rall)
        average_reward.append(rall)
        print("Episode:{0:6d} | Frames:{1:9d} | Steps:{2:5d} | Reward:{3:3.0f} | e-greedy:{4:.5f} | Avg_Max_Q:{5:2.5f} | "
              "Recent reward:{6:.5f}  ".format(episode,frame, count, rall, e, avg_max_Q/float(count),np.mean(recent_rlist)))