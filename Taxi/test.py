import numpy as np
import gym
import random
import sys
import cvxpy as cp

N_idx = 20
F_idx = 4
GAMMA = 0.99


if __name__ == '__main__':
    print(":: Testing Taxi APP-learning.\n")
    
    # Load the agent
    n_states = 500
    n_actions = 6
    q_table = np.load(file="learner_q_table_art.npy")

    # Create a new game instance.
    env = gym.make('Taxi-v2')
    n_episode = 1 # test the agent 10times
    scores = []
    episode_actions = []

    for ep in range(n_episode):
        state = env.reset()
        score = 0
        actions = []
        while True:
            # Render the play
            env.render()

            act_set = np.where(q_table[state]==np.max(q_table[state]))[0]
            action = act_set[np.random.randint(len(act_set))]

            action = np.argmax(q_table[state])
            actions.append(action)

            next_state, reward, done, _ = env.step(action)
            
            score += reward
            scores.append(reward)
            state = next_state

            if done:
                env.render()
                print('{} episode | score: {:.1f}'.format(ep + 1, score))                
                break

        episode_actions.append(actions)

    env.close()
#sys.exit()
    print(episode_actions)
    print(scores)