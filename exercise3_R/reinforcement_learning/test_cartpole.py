import os
from datetime import datetime
import gym
import json
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import *
import numpy as np
import time
import argparse


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

    env = gym.make("CartPole-v0").unwrapped
    state_dim = 4
    num_actions = 2
    hidden_dim = 400

    Q = MLP(state_dim, num_actions, hidden_dim)
    Q.eval()
    agent = DQNAgent(Q, Q, num_actions)
    # loading weights
    agent.load(model)

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False,
                            rendering=rendering, max_timesteps=max_timesteps)
        episode_rewards.append(stats.episode_reward)
        time.sleep(1)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
    print("Mean: {}; Std. Dev.: {}".format(results["mean"], results["std"]))
