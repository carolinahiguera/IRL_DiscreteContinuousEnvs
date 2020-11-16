import numpy as np
import cvxpy as cp

class FeatureEstimate:
    def __init__(self, env):
        self.env = env     
        self.feature_num = 3   
        self.feature = np.ones(self.feature_num)    

    def manhattan_distance(self,x,y):
        return np.abs(x[0]-y[0])+np.abs(x[1]-y[1])

    def get_features(self, observation):
        taxi = observation[0]
        passenger = observation[1]
        destination = observation[2]
        time = observation[3]
        # position passenge
        distance = self.manhattan_distance(taxi, passenger)
        self.feature[0] = 1.0 - distance**2

        # position destination
        distance = self.manhattan_distance(taxi, destination)
        self.feature[1] = 1.0 - distance**2  

        # time        
        self.feature[2] = - (time**2)

        return self.feature

def expert_feature_expectation(gamma, demonstrations, env):
    feature_estimate = FeatureEstimate(env)
    feature_expectations = np.zeros(feature_estimate.feature_num)
    locs = [(0,0), (0,4), (4,0), (4,3)]
    
    for demo_num in range(len(demonstrations)):
        sample = demonstrations[demo_num]
        for demo_length in range(len(demonstrations[demo_num])):
            taxi = (sample[demo_length][1], sample[demo_length][2])
            passenger = locs[sample[demo_length][3]] if sample[demo_length][3]<4 else taxi
            destination = locs[sample[demo_length][4]]
            observation = [taxi, passenger, destination, demo_length]
            features = feature_estimate.get_features(observation)
            feature_expectations += (gamma**(demo_length)) * np.array(features)
    
    feature_expectations = feature_expectations / len(demonstrations)
    
    return feature_expectations

def calc_feature_expectation(gamma, q_table, demonstrations, env):
    feature_estimate = FeatureEstimate(env)
    feature_expectations = np.zeros(feature_estimate.feature_num)
    demo_num = len(demonstrations)
    locs = [(0,0), (0,4), (4,0), (4,3)]
    
    for _ in range(demo_num):
        state = env.reset()
        demo_length = 0
        done = False
        
        while not done:
            demo_length += 1

            action = np.argmax(q_table[state])
            next_state, reward, done, _ = env.step(action)

            position = []
            for pos in env.decode(next_state):
                position.append(pos)
            taxi = (position[0], position[1])
            passenger = locs[position[2]] if position[2]<4 else taxi
            destination = locs[position[3]]
            observation = [taxi, passenger, destination, demo_length]
            
            features = feature_estimate.get_features(observation)
            feature_expectations += (gamma**(demo_length)) * np.array(features)

            state = next_state
    
    feature_expectations = feature_expectations/ demo_num

    return feature_expectations

def QP_optimizer(feature_num, learner, expert):
    w = cp.Variable(feature_num)
    
    obj_func = cp.Minimize(cp.norm(w))
    constraints = [(expert-learner) * w >= 2] 

    prob = cp.Problem(obj_func, constraints)
    prob.solve()

    if prob.status == "optimal":
        print("status:", prob.status)
        print("optimal value", prob.value)
    
        weights = np.squeeze(np.asarray(w.value))
        return weights, prob.status
    else:
        print("status:", prob.status)
        
        weights = np.zeros(feature_num)
        return weights, prob.status

def add_feature_expectation(learner, temp_learner):
    # save new feature expectation to list after RL step
    learner = np.vstack([learner, temp_learner])
    return learner

def subtract_feature_expectation(learner):
    # if status is infeasible, subtract first feature expectation
    learner = learner[1:][:]
    return learner