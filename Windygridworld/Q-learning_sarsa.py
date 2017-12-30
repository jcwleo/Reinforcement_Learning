from windygridworld import WindyGridWorld
import numpy as np
import matplotlib.pyplot as plt
import random as rn


def rargmax(vector):
    '''모두 같은 Q값일 때, 랜덤하게 액션을 정해주는 함수

    Args:
        vector(ndarray): Q-table

    Returns:
        action: 랜덤하게 정해진 action값

    '''
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rn.choice(indices)


def array2index(array, width):
    '''

    Args:
        array: gridworld array
        width: gridworld의 너비

    Returns:
        idx: 2D array인 array를 인덱스 값으로 바꾼 값
    '''
    idx = array[0] * width + array[1]
    return idx


def learning(max_step, learning_type,render):
    env = WindyGridWorld()

    # Q-table 생성
    Q = np.zeros([env.observation_space, env.action_space])

    global_step = 0

    # 하이퍼파라미터
    alpha = 0.5
    epsilon = 0.1

    episode = 0
    plot_graph = []

    while global_step <= max_step:
        episode += 1

        # 에피소드 리셋
        state = env.reset()

        done = False
        step = 0
        total_reward = 0
        while not done:
            if render:
                env.render()

            step += 1
            global_step += 1
            plot_graph.append(episode)

            # e-greedy 방법으로 action 결정
            if epsilon > np.random.rand(1):
                action = np.random.randint(env.action_space)
            else:
                action = rargmax(Q[array2index(state, env.width), :])

            # 실제 액션 수행 <next state, reward, terminal, info>
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            # Q-learning일 때와 SARSA일 때를 구별하여 학습
            if learning_type == 'Q-Learning':
                # Q-learning
                # Q(s,a) = Q(s,a) + a * (reward + max_a(Q(s',a)) - Q(s,a))
                Q[array2index(state, env.width), action] += (
                    alpha * (reward + np.max(Q[array2index(next_state, env.width), :])
                             - Q[array2index(state, env.width), action]))
            else:
                # SARSA
                # Q(s,a) = Q(s,a) + a * (reward + Q(s',a') - Q(s,a))
                Q[array2index(state, env.width), action] += (
                    alpha * (reward + (Q[array2index(next_state, env.width), action])
                             - Q[array2index(state, env.width), action]))

            state = next_state[:]

        print('Learning Type : {}   Episode : {:5.0f}  Step : {:5.0f}  reward : {:5.0f}'
              .format(learning_type,episode,step,total_reward))

    # 학습된 Q값 저장
    np.save('QValue/{}_value'.format(learning_type), Q)
    np.savetxt('QValue/{}_value.txt'.format(learning_type), Q)

    direction = np.array(['L', 'U', 'R', 'D'])

    # 학습된 Optimal한 action 추출
    Q = np.argmax(Q, axis=1)
    optimal_policy = np.chararray([env.observation_space], unicode=True)
    for i in range(env.action_space):
        optimal_policy[Q == i] = direction[i]

    optimal_policy = optimal_policy.reshape([env.height, env.width])

    # Optimal policy를 txt로 저장
    np.savetxt('OptimalPolicy/optimal_{}.txt'.format(learning_type), optimal_policy, delimiter='', fmt='%s')

    return plot_graph


def main():
    # 학습시킬 step 수
    max_step = 20000

    # 움직임을 실제 보고싶을시 True로 변경
    render = False

    # 각각 Q_learning과 Sarsa 학습
    q_graph = learning(max_step, 'Q-Learning',render)
    sarsa_graph = learning(max_step, 'SARSA', render)

    # Q_learning과 Sarsa 학습 그래프 Plot
    plt.xlim([0, max_step * 1.1])
    plt.plot(q_graph, 'b', label='Q-learning')
    plt.plot(sarsa_graph, 'g', label='SARSA')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('graph.png')
    plt.show()


if __name__ == '__main__':
    main()
