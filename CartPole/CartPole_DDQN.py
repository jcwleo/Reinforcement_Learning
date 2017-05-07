# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
from gym import wrappers
import numpy as np
import random as ran

env = gym.make('CartPole-v0')

# 꺼내서 사용할 리플레이 갯수
REPLAY = 10
# 리플레이를 저장할 리스트
REPLAY_MEMORY = []
# 미니배치
MINIBATCH = 50

INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n

# 하이퍼파라미터
LEARNING_LATE = 0.001
DISCOUNT = 0.99
model_path = "save/model.ckpt"


# 두개의 네트워크 구성

x=tf.placeholder(dtype=tf.float32, shape=(None, INPUT))

y=tf.placeholder(dtype=tf.float32, shape=(None, OUTPUT))
dropout = tf.placeholder(dtype=tf.float32)

# Main 네트워크
W1 = tf.get_variable('W1',shape=[INPUT, 200],initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('W2',shape=[200,200],initializer=tf.contrib.layers.xavier_initializer())
# W3 = tf.get_variable('W3',shape=[200,150],initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable('W4',shape=[200, OUTPUT],initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros([1],dtype=tf.float32))
b2 = tf.Variable(tf.zeros([1],dtype=tf.float32))

_L1=tf.nn.relu(tf.matmul(x,W1)+b1)
L1=tf.nn.dropout(_L1,dropout)
_L2=tf.nn.relu(tf.matmul(L1,W2)+b2)
L2=tf.nn.dropout(_L2,dropout)
# L3=tf.nn.relu(tf.matmul(L2,W3))
Q_pre = tf.matmul(L2,W4)

# Target 네트워크
W1_r = tf.get_variable('W1_r',shape=[INPUT, 200])
W2_r = tf.get_variable('W2_r',shape=[200,200])
# W3_r = tf.get_variable('W3_r',shape=[200,150])
W4_r = tf.get_variable('W4_r',shape=[200, OUTPUT])

b1_r = tf.Variable(tf.zeros([1],dtype=tf.float32))
b2_r = tf.Variable(tf.zeros([1],dtype=tf.float32))


L1_r=tf.nn.relu(tf.matmul(x ,W1_r)+b1_r)
L2_r=tf.nn.relu(tf.matmul(L1_r,W2_r)+b2_r)
# L3_r=tf.nn.relu(tf.matmul(L2_r,W3_r))
Q_pre_r = tf.matmul(L2_r,W4_r)

# 총 Reward를 저장해놓을 리스트
rlist=[0]
recent_rlist=[0]

episode = 0

# Loss function 정의
cost = tf.reduce_sum(tf.square(y-Q_pre))
optimizer = tf.train.AdamOptimizer(LEARNING_LATE, epsilon=0.01)
train = optimizer.minimize(cost)


saver = tf.train.Saver()

# 세션 정의
with tf.Session(config = tf.ConfigProto(device_count ={'GPU' : 0})) as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())
    # Target 네트워크에 main 네트워크 값을 카피해줌
    sess.run(W1_r.assign(W1))
    sess.run(W2_r.assign(W2))
    sess.run(W4_r.assign(W4))
    sess.run(b1_r.assign(b1))
    sess.run(b2_r.assign(b2))

    # 에피소드 시작
    while np.mean(recent_rlist) < 195 :
        episode += 1

        # state 초기화
        s = env.reset()
        if len(recent_rlist) > 200:
            del recent_rlist[0]
        # e-greedy
        e = 1. / ((episode/25)+1)

        rall = 0
        d = False
        count = 0

        # 에피소드가 끝나기 전까지 반복
        while not d and count < 10000 :

            #env.render()
            count += 1

            # state 값의 전처리
            s_t = np.reshape(s,[1,INPUT])

            # 현재 상태의 Q값을 에측
            Q = sess.run(Q_pre, feed_dict={x:s_t, dropout: 1})

            # e-greedy 정책으로 랜덤하게 action 결정
            if e > np.random.rand(1):
                a = env.action_space.sample()
            else:
                a = np.argmax(Q)

            # 결정된 action으로 Environment에 입력
            s1, r, d, _ = env.step(a)

            # Environment에서 반환한 Next_state, action, reward, done 값들을
            # Replay_memory에 저장
            REPLAY_MEMORY.append([s_t,a,r,s1,d,count])

            # 저장된 값들이 50000개 이상 넘어가면 맨 앞 Replay부터 삭제
            if len(REPLAY_MEMORY) > 50000:
                del REPLAY_MEMORY[0]

            # 총 reward 합
            rall += r
            # state를 Next_state로 바꿈
            s = s1


        # 10번의 episode마다 학습
        if len(REPLAY_MEMORY) > 50:

            # 50번의 미니배치로 학습
                # 저장된 리플레이 중에 학습에 사용할 랜덤한 리플레이 샘플들을 가져옴
            for sample in ran.sample(REPLAY_MEMORY, REPLAY):

                s_t_r, a_r, r_r, s1_r, d_r ,count_r= sample

                # 꺼내온 리플레이의 state의 Q값을 예측
                Y = sess.run(Q_pre, feed_dict={x: s_t_r, dropout: 1})

                if d_r:
                    # 꺼내온 리플레이의 상태가 끝난 상황이라면 Negative Reward를 부여
                    if count_r < env.spec.timestep_limit :
                        Y[0, a_r] = -100
                else:
                    # 끝나지 않았다면 Q값을 업데이트
                    s1_t_r= np.reshape(s1_r,[1,INPUT])
                    Q1, Q = sess.run([Q_pre_r,Q_pre], feed_dict={x: s1_t_r, dropout:1})
                    Y[0, a_r] = r_r + DISCOUNT * Q1[0, np.argmax(Q)]

                # 업데이트 된 Q값으로 main네트워크를 학습
                _, loss = sess.run([train, cost], feed_dict={x: s_t_r, y: Y, dropout:1})

            # 10번 마다 target 네트워크에 main 네트워크 값을 copy
            sess.run(W1_r.assign(W1))
            sess.run(W2_r.assign(W2))
            sess.run(W4_r.assign(W4))
            sess.run(b1_r.assign(b1))
            sess.run(b2_r.assign(b2))
            print(loss)

        # 총 reward의 합을 list에 저장
        recent_rlist.append(rall)
        rlist.append(rall)
        print("Episode:{} steps:{} reward:{} average reward:{} recent reward:{}".format(episode, count, rall,
                                                                                        np.mean(rlist),
                                                                                        np.mean(recent_rlist)))

    save_path = saver.save(sess, model_path)
    print("Model saved in file: ",save_path)


    rlist=[]
    recent_rlist=[]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)

    print("Model restored form file: ", save_path)
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
            Q = sess.run(Q_pre, feed_dict={x: s_t,dropout: 1})
            a = np.argmax(Q)

            # 결정된 action으로 Environment에 입력
            s, r, d, _ = env.step(a)

            # 총 reward 합
            rall += r


        rlist.append(rall)

        print("Episode : {} steps : {} r={}. averge reward : {}".format(episode, count, rall,
                                                                        np.mean(rlist)))

