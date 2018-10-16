# Reinforcement Learning
###### 여러 환경에 적용해보는 강화학습 예제(파이토치로 옮기고 있습니다)
# [Here is my new Repo for Policy Gradient!!](https://github.com/jcwleo/mario_rl)
-------------------

![Alt text](/readme/Play.gif)
###### [Breakout / Use DQN(Nature2015)]

---------------
## 1. Q-Learning / SARSA
* FrozenLake(Gridword)
	* [Deterministic Q-Learning](https://github.com/jcwleo/Reinforcement_Learning/blob/master/FrozenLake/FL_Q-Table.py)
	* [Add Exploration & Discounted Factor](https://github.com/jcwleo/Reinforcement_Learning/blob/master/FrozenLake/FL_Q-table_exp%26dis.py)
	* [Stochastic Q-Learning](https://github.com/jcwleo/Reinforcement_Learning/blob/master/FrozenLake/FL_Q-table_Stochastic.py)
* WindyGridWorld(in Sutton's book)
    * [Q-Learning / SARSA](https://github.com/jcwleo/Reinforcement_Learning/tree/master/Windygridworld)
## 2. Q-Network (Action-Value Function Approximation)
* [FrozenLake(Gridword)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/FrozenLake/FrozenLake_Q-Network.py)
* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_Q-Network.py)

## 3. DQN
DQN(NIPS2013)은 (Experience Replay Memory / CNN) 을 사용.
* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_DQN_NIPS2013.py) - Cartpole 같은 경우에는 CNN을 사용하지 않고 센서 정보를 통해서 학습

DQN(Nature2015)은 (Experience Replay Memory / Target Network / CNN) 을 사용

* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_DQN_Nature2015.py)
* [Breakout(atari)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/Breakout/Breakout_DQN_class.py)
* [Breakout(atari)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/Breakout/breakout_dqn_pytorch.py)
	* this code is made by pytorch and more efficient memory and train

## 5. Vanilla Policy Gradient(REINFORCE)
* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_PolicyGradient.py)
* [Pong(atari)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/Pong/Pong_PolicyGradient.py)
* [Breakout(atari)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/Breakout/Breakout_PolicyGradient.py)

## 6. Advantage Actor Critic
* episodic
	* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_A2C_episodic.py)
	* [Pong(atari)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/Pong/Pong_A2C_episodic.py)
* one-step
    *  [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/Cartpole_A2C_onestep.py)
* n-step
    * [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/Cartpole_A2C_nstep.py)

## 7. Deep Deterministic Policy Gradient
   * [Pendulum(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/pendulum/pendulum_ddpg.py)

## 8. Parallel Advantage Actor Critic(is called 'A2C' in OpenAI)
* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_PAAC.py)(used a single thread instead of multi thread)
* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_PAAC_multiproc.py)(used multiprocessing in pytorch)
* [Super Mario Bros](https://github.com/jcwleo/mario_rl)(used multiprocessing in pytorch)

## 9. C51(Distributional RL)
* DDQN
	* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_C51.py)
	
## 10. PPO(Proximal Policy Optimization)
* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/cartpole_ppo.py)
