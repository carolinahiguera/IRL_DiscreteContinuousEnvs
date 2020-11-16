import gym
import numpy as np
import readchar


# MACROS
south = 0
north = 1
east = 2
west = 3
pickup = 4 
dropoff = 5 

# Key mapping
arrow_keys = {
    '\x1b[A': north,
    '\x1b[B': south,
    '\x1b[C': east,
    '\x1b[D': west,
    'a': pickup,
    's': dropoff}

env = gym.make('Taxi-v2')
trajectories = []
NUM_TRAJECTORIES = 20

for trajectory_idx in range(NUM_TRAJECTORIES): # n_trajectories : 20
    trajectory = []
    step = 0
    done = False
    env.reset()
    print(f'Trajectory: {trajectory_idx}')

    while not done:
        env.render()
        print("step", step)

        key = readchar.readkey()
        if key not in arrow_keys.keys():
            break

        action = arrow_keys[key]
        state, reward, done, _ = env.step(action)
        position = []
        for pos in env.decode(state):
            position.append(pos)

        #discrete state, taxi_row, taxi_col, pass_loc, dest_idx
        trajectory.append((state, position[0], position[1], position[2],
                            position[3], action, reward))
        step += 1

    trajectory_numpy = np.array(trajectory, float)
    print("trajectory_numpy.shape", trajectory_numpy.shape)    
    trajectories.append(trajectory)
    print('')

np_trajectories = np.array(trajectories, float)
print("np_trajectories.shape", np_trajectories.shape)

np.save("expert_taxi", arr=np_trajectories)