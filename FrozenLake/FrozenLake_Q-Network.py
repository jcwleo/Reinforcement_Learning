import tensorflow as tf
import gym
import numpy as np

env = gym.make('FrozenLake-v0')

x=tf.placeholder(dtype=tf.float32, shape=(1,env.observation_space.n))

W1=tf.Variable(tf.random_uniform((env.observation_space.n, env.action_space.n),-0.1,0.1))
Q_pre = tf.matmul(x,W1)


y=tf.placeholder(dtype=tf.float32, shape=(1, env.action_space.n))



learning_rate = 0.1
num_episode = 2000
e = 0.1
discount_factor = 0.99
rlist=[]
slist=[]

cost = tf.reduce_sum(tf.square(y-Q_pre))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()


def one_hot(x):
    return np.identity(env.observation_space.n)[x:x + 1]


with tf.Session() as sess:
    sess.run(init)
    for step in range(num_episode):

        s = env.reset()
        e = 1. / ((step/50)+10)
        rall = 0
        d = False
        j=0
        while not d:
            j+=1
            Q = sess.run(Q_pre, feed_dict={x: one_hot(s)})

            if e > np.random.rand(1):
                a = env.action_space.sample()
            else:
                a = np.argmax(Q)

            s1, r, d, _ = env.step(a)

            if d:
                Q[0, a] = r
            else:
                Q1 = sess.run(Q_pre, feed_dict={x: one_hot(s1)})
                Q[0, a] = r + discount_factor * np.max(Q1)

            sess.run(train, feed_dict={x: one_hot(s), y: Q})

            rall += r
            slist.append(s)
            s = s1
        print(slist)
        slist=[]
        rlist.append(rall)
        print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(step, j, rall, np.mean(rlist)))

print("성공한 확률" + str(sum(rlist) / num_episode) + "%")

