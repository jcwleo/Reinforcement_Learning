# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
import numpy as np
import random as ran

env = gym.make('CartPole-v1')

# 꺼내서 사용할 리플레이 갯수
REPLAY = 10
# 리플레이를 저장할 리스트
REPLAY_MEMORY = []
# 미니배치
MINIBATCH = 50

INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n

# 하이퍼파라미터
LEARNING_LATE = 0.01
NUM_EPISODE = 2000

DISCOUNT = 0.99


# 네트워크 구성
x=tf.placeholder(dtype=tf.float32, shape=(1,4))

W1 = tf.get_variable('W1',shape=[INPUT,10],initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('W4',shape=[10, OUTPUT],initializer=tf.contrib.layers.xavier_initializer())

L1=tf.nn.tanh(tf.matmul(x,W1))
Q_pre = tf.matmul(L1,W2)

y=tf.placeholder(dtype=tf.float32, shape=(1, env.action_space.n))

# 손실 함수
loss = tf.reduce_sum(tf.square(y-Q_pre))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_LATE)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

rList=[]

with tf.Session() as sess:
    sess.run(init)
    for episode in range(5000):

        s = env.reset()

        e = 1. / ((episode/25)+1)
        rall = 0
        d = False
        count=0

        while not d:
            # env.render()
            count+=1

            # 현재 상태(s)로 Q값을 예측
            s_t = np.reshape(s,[1,INPUT])
            Q = sess.run(Q_pre, feed_dict={x:s_t})

            # e-greedy 를 사용하여 action값 구함
            if e > np.random.rand(1):
                a = env.action_space.sample()
            else:
                a = np.argmax(Q)

            # action을 취함
            s1, r, d, _ = env.step(a)

            # state, action, reward, next_state, done 을 메모리에 저장
            REPLAY_MEMORY.append([s_t,a,r,s1,d])

            # 메모리에 50000개 이상의 값이 들어가면 가장 먼저 들어간 것부터 삭제
            if len(REPLAY_MEMORY) > 50000:
                del REPLAY_MEMORY[0]

            rall += r
            s = s1

        # 10 번의 스탭마다 미니배치로 학습
        if episode % 10 == 1 :

            for i in range(MINIBATCH):

                # 메모리에서 사용할 리플레이를 랜덤하게 가져옴
                for sample in ran.sample(REPLAY_MEMORY, REPLAY):

                    s_t_r, a_r, r_r, s1_r ,d_r = sample

                    # DQN 알고리즘으로 학습
                    if d_r:
                        Q[0, a_r] = -100
                    else:
                        s1_t_r= np.reshape(s1_r,[1,INPUT])

                        Q1 = sess.run(Q_pre, feed_dict={x: s1_t_r})

                        Q[0, a_r] = r_r + DISCOUNT * np.max(Q1)

                    sess.run(train, feed_dict={x: s_t_r, y: Q})



        rList.append(rall)
        print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(episode, count, rall, np.mean(rList)))


    for episode in range(500):
        # state 초기화
        s = env.reset()

        rall = 0
        d = False
        count = 0
        # 에피소드가 끝나기 전까지 반복
        while not d :
            env.render()
            count += 1
            # state 값의 전처리
            s_t = np.reshape(s, [1, INPUT])

            # 현재 상태의 Q값을 에측
            Q = sess.run(Q_pre, feed_dict={x: s_t})
            a = np.argmax(Q)

            # 결정된 action으로 Environment에 입력
            s, r, d, _ = env.step(a)

            # 총 reward 합
            rall += r


        rList.append(rall)

        print("Episode : {} steps : {} r={}. averge reward : {}".format(episode, count, rall,
                                                                        np.mean(rList)))




