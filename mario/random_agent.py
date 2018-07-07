import gym_super_mario_bros
env = gym_super_mario_bros.make('SuperMarioBros-v0')

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())

env.close()