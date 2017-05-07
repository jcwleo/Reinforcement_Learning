# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
import numpy as np

env = gym.make('CartPole-v0')

# 네트워크 구성

x=tf.placeholder(dtype=tf.float32, shape=(1,4))

input = env.observation_space.shape[0]

W1=tf.get_variable('W1',shape=[input,10],initializer=tf.contrib.layers.xavier_initializer())
W2=tf.get_variable('W2',shape=[10,20],initializer=tf.contrib.layers.xavier_initializer())
W3=tf.get_variable('W3',shape=[20,15],initializer=tf.contrib.layers.xavier_initializer())
W4=tf.get_variable('W4',shape=[15,env.action_space.n],initializer=tf.contrib.layers.xavier_initializer())


L1=tf.nn.relu(tf.matmul(x,W1))
L2=tf.nn.relu(tf.matmul(L1,W2))
L3=tf.nn.relu(tf.matmul(L2,W3))
Q_pre = tf.matmul(L3,W4)


y=tf.placeholder(dtype=tf.float32, shape=(1, env.action_space.n))

# 하이퍼 파라미터 정의
learning_rate = 0.1
num_episode = 2000
e = 0.1
discount_factor = 0.99
rlist=[]

# 손실 함수 정의
cost = tf.reduce_sum(tf.square(y-Q_pre))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 변수 초기화
    sess.run(init)
    for step in range(num_episode):
        # stats 초기화
        s = env.reset()
        # e-greedy
        e = 1. / ((step/50)+10)
        rall = 0
        d = False
        j=0
        s_t = sess.run(tf.expand_dims(s, axis=0))
        while not d:
            # env.render()
            j+=1

            # reshape을 통한 state 전처리

            # 현재 state에 대한 Q값 예측
            Q = sess.run(Q_pre, feed_dict={x:s_t})

            # e-greedy 를 통한 랜덤한 action
            if e > np.random.rand(1):
                a = env.action_space.sample()
            else:
                a = np.argmax(Q)

            # action 수행
            s1, r, d, _ = env.step(a)


            if d:
                # 에피소드가 끝났을때 Negative reward 부여
                Q[0, a] = -100
            else:
                # next_state값의 전처리 후 Q-learning
                s1_t = sess.run(tf.expand_dims(s1, axis=0))
                Q1 = sess.run(Q_pre, feed_dict={x: s1_t})
                Q[0, a] = r + discount_factor * np.max(Q1)

            sess.run(train, feed_dict={x: s_t, y: Q})

            rall += r

            s_t = s1_t

        slist=[]
        rlist.append(rall)
        print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(step, j, rall, np.mean(rlist)))

print("성공한 확률" + str(sum(rlist) / num_episode) + "%")

