from __future__ import print_function

import gym
from agent.dqn_agent import DQNAgent
from train_carracing import run_episode
from agent.networks import *
import numpy as np
import argparse
import os
from datetime import datetime
import json


np.random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to test cartpole with a model.')
    parser.add_argument('-n', '--num_episodes', dest='n_test_episodes', type=int, default=200,
                        help='Number of episodes to run.')
    parser.add_argument('-r', '--rendering', dest='rendering', type=str, default='True',
                        choices=['True', 'False'], help='Number of episodes to evaluate.')
    parser.add_argument('-t', '--timesteps', dest='max_timesteps', type=int, default=200,
                        help='Maximum timesteps for an episode.')
    parser.add_argument('-m', '--model', dest='model', type=str,
                        help='Path to model.')
    args = parser.parse_args()
    n_test_episodes = args.n_test_episodes
    max_timesteps = args.max_timesteps
    model = args.model
    rendering = True if args.rendering=='True' else False

    env = gym.make("CarRacing-v0").unwrapped

    history_length =  0

    #TODO: Define networks and load agent
    # ....

    # n_test_episodes = 15

    num_actions = 5
    Q = CNN()
    Q.eval()

    mean = []
    std = []

    model_list = os.listdir(model) #[20:]
    print(model_list[0])
    for i in range(len(model_list)):
        print("{}/{}".format(i, len(model_list)))
        agent = DQNAgent(Q, Q, num_actions)
		# loading weights
        agent.load(os.path.join(model, model_list[i]))

        episode_rewards = []
        for i in range(n_test_episodes):
            stats = run_episode(env, agent, deterministic=True, do_training=False,
		                        skip_frames=10, rendering=rendering,
                                max_timesteps=max_timesteps, history_length=0, eps=0.1)
            episode_rewards.append(stats.episode_reward)
            print(episode_rewards[-1])
        mean.append(np.mean(episode_rewards))
        std.append(np.std(episode_rewards))
        print(mean[-1], std[-1])
        print()

    print(mean)
    print(std)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["mean"] = np.array(mean).tolist()
    results["std"] = np.array(std).tolist()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/local_carracing_evaluation_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
