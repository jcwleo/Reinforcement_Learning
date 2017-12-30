# Reinforcement Learning
###### 여러 환경에 적용해보는 강화학습 예제

![Alt text](/readme/Play.gif)
###### [Breakout / Use DQN(Nature2015)]
![Alt text](/readme/1x1conv.gif)
###### [Breakout / Image from the first conv layer(1x1)]
---------------
## 1. Q-Learning / SARSA
* FrozenLake(Gridword)
	* [Deterministic](https://github.com/jcwleo/Reinforcement_Learning/blob/master/FrozenLake/FL_Q-Table.py)
	* [Exploration & Discounted Factor](https://github.com/jcwleo/Reinforcement_Learning/blob/master/FrozenLake/FL_Q-table_exp%26dis.py)
	* [Stochastic](https://github.com/jcwleo/Reinforcement_Learning/blob/master/FrozenLake/FL_Q-table_Stochastic.py)
* WindyGridWorld(in Sutton's book)
    * [Q-Learning / SARSA](https://github.com/jcwleo/Reinforcement_Learning/tree/master/Windygridworld)
## 2. Q-Network (just use network)
* [FrozenLake(Gridword)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/FrozenLake/FrozenLake_Q-Network.py)
* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_Q-Network.py)

## 3. DQN(NIPS2013)
DQN(NIPS2013)은 (Experience Replay Memory / CNN) 을 사용.
* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_DQN_NIPS2013.py) - Cartpole 같은 경우에는 CNN을 사용하지 않고 센서 정보를 통해서 학습

## 4. DQN(Nature2015)
DQN(Nature2015)은 (Experience Replay Memory / Target Network / CNN) 을 사용

* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_DQN_Nature2015.py) - Cartpole 같은 경우에는 CNN을 사용하지 않고 센서 정보를 통해서 학습
* [Breakout(atari)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/Breakout/Breakout_DQN_class.py)

## 5. Policy Gradient
* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_PolicyGradient.py) - Cartpole 같은 경우에는 CNN을 사용하지 않고 센서 정보를 통해서 학습
* [Pong(atari)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/Pong/Pong_PolicyGradient.py)
* [Breakout(atari)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/Breakout/Breakout_PolicyGradient.py)

## 6. Advantage Actor Critic
* episodic
	* [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/CartPole_A2C_episodic.py) - Cartpole 같은 경우에는 CNN을 사용하지 않고 센서 정보를 통해서 학습
	* [Pong(atari)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/Pong/Pong_A2C_episodic.py)
* one-step
    *  [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/Cartpole_A2C_onestep.py) - Cartpole 같은 경우에는 CNN을 사용하지 않고 센서 정보를 통해서 학습
* n-step
    * [CartPole(Classic Control)](https://github.com/jcwleo/Reinforcement_Learning/blob/master/CartPole/Cartpole_A2C_nstep.py) - Cartpole 같은 경우에는 CNN을 사용하지 않고 센서 정보를 통해서 학습

## 7. PAAC(Parallel Advantage Actor Critic)
* Work In Progress