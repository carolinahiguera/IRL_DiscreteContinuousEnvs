import sys
import gym
import pylab
import numpy as np
from app import *

n_states = 500 
n_actions = 6
q_table = np.zeros((n_states, n_actions))

gamma = 0.99
q_learning_rate = 0.03

def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action]
    q_2 = reward + gamma * np.max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)

def main():
    locs = [(0,0), (0,4), (4,0), (4,3)]
    env = gym.make('Taxi-v2')
    demonstrations = np.load(file="expert_taxi.npy")

    feature_estimate = FeatureEstimate(env)
    
    learner = calc_feature_expectation(gamma, q_table, demonstrations, env)
    learner = np.matrix([learner])
    
    expert = expert_feature_expectation(gamma, demonstrations, env)
    expert = np.matrix([expert])
    
    w, status = QP_optimizer(feature_estimate.feature_num, learner, expert)

    episodes, scores = [], []
    
    for episode in range(60000):
        state = env.reset()
        score = 0
        steps = 0.0
        while True:            
            act_set = np.where(q_table[state]==np.max(q_table[state]))[0]
            action = act_set[np.random.randint(len(act_set))]

            next_state, reward, done, _ = env.step(action)
            
            position = []
            for pos in env.decode(next_state):
                position.append(pos)
            taxi = (position[0], position[1])
            passenger = locs[position[2]] if position[2]<4 else taxi
            destination = locs[position[3]]
            observation = [taxi, passenger, destination, steps]

            features = feature_estimate.get_features(observation)
            irl_reward = np.dot(w, features)
            
            update_q_table(state, action, irl_reward, next_state)

            score += reward
            steps += 1.0
            state = next_state

            if done:
                scores.append(score)
                episodes.append(episode)
                break

        if episode % 1000 == 0:
            score_avg = np.mean(scores)
            print('{} episode score is {:.2f}'.format(episode, score_avg))
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("app_eps_60000_art.png")
            np.save("learner_q_table_art", arr=q_table)

        if episode % 5000 == 0:
            # optimize weight per 5000 episode
            status = "infeasible"
            temp_learner = calc_feature_expectation(gamma, q_table, demonstrations, env)
            learner = add_feature_expectation(learner, temp_learner)
            
            while status=="infeasible":
                w, status = QP_optimizer(feature_estimate.feature_num, learner, expert)
                if status=="infeasible":
                    learner = subtract_feature_expectation(learner)

if __name__ == '__main__':
    main()
