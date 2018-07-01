import gym_super_mario_bros
import numpy as np
env = gym_super_mario_bros.make('SuperMarioBros-v2')
print(env.action_space.n)
done = True
for step in range(5000):
    if done:
        print(step)
        state = env.reset()
    if step < 30:
        state, reward, done, info = env.step(2)
    else:
        state, reward, done, info = env.step(1)
    reward = np.clip(reward, -1, 1)
    print(reward)

env.close()