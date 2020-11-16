# IRL_DiscreteContinuousEnvs

![](test_taxi.eps)

Inverse Reinforcement Learning to the Taxi-v2 gym environment of OpenAi.


## Make and expert agent

With this script you cand be the taxi driver expert and extract the trajectories in the expert_taxi.npy file.

The actions are:

```json
	"up": "key-arrow up",
	"down": "key-arrow down",
	"left": "key-arrow left",
	"right": "key-arrow right",
	"pickup": "a",
	"dropoff": "s"

To run the script:
```
python makeExpert.py
```
## Training an agent with IRL
To run the script:
```
python train.py
```

## Testing the policy
To run the script:
```
python test.py
```
