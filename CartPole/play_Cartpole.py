import tensorflow as tf
import gym
import numpy as np

env = gym.make("CartPole-v0")

print(env.observation_space)
INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n

# 하이퍼파라미터
LEARNING_LATE = 0.001
DISCOUNT = 0.99

# Main 네트워크
x=tf.placeholder(dtype=tf.float32, shape=(None, INPUT))

y=tf.placeholder(dtype=tf.float32, shape=(None, OUTPUT))
dropout = tf.placeholder(dtype=tf.float32)

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

saver = tf.train.Saver()
model_path = "save/model.ckpt"
with tf.Session() as sess:
    rlist=[]
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)

    print("Model restored form file: ", model_path)
    for episode in range(500):
        # state 초기화
        s = env.reset()
        e= 0.1
        rall = 0
        d = False
        count = 0
        # 에피소드가 끝나기 전까지 반복
        while not d and count < 5000:
            env.render()
            count += 1
            # state 값의 전처리
            s_t = np.reshape(s, [1, INPUT])

            # 현재 상태의 Q값을 에측
            Q = sess.run(Q_pre, feed_dict={x: s_t, dropout: 1})

            if e > np.random.rand(1):
                a = env.action_space.sample()
            else:
                a = np.argmax(Q)


            # 결정된 action으로 Environment에 입력

            s, r, d, _ = env.step(a)

            # 총 reward 합
            rall += r

        rlist.append(rall)

        print("Episode : {} steps : {} r={}. averge reward : {}".format(episode, count, rall,
                                                                        np.mean(rlist)))

