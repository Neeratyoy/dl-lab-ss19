from __future__ import print_function

import sys
sys.path.append("../")

import argparse
from datetime import datetime
import numpy as np
import gym
import os
import json

from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, rendering=True, max_timesteps=1000, history_length=1):

    episode_reward = 0
    step = 0

    state = rgb2gray(env.reset())
    state = np.array([state])
    # input_state = np.random.randint(0, 255, (1, history_length, state.shape[-2], state.shape[-1]))
    # input_state[0][0]

    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events()
    i = 1
    pause = 0
    while True:

        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        #    state = ...


        while i < history_length:
            next_state, _, done, _ = env.step(id_to_action(0))
            next_state = rgb2gray(next_state)
            state = np.append(state, [next_state], axis=0)
            i = i+1

        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...
        # when i >= history_length

        # a = id_to_action(3)
        if np.all(next_state == state):
            pause = pause + 1
            if pause > 5:
                push_start = 20
                print("Pushing the car! Get moving!!")
                while push_start > 0:
                    a = id_to_action(3)
                    push_start = push_start - 1
                    next_state, r, done, info = env.step(a)
                    next_state = rgb2gray(next_state)
                    episode_reward += r
                    state = np.append(state[1:], [next_state], axis=0)
                    step += 1
                pause = 0
        a, _ = agent.predict(np.array([state]))
        # print(a)
        a = np.argmax(a.detach().cpu().numpy()[0])
        a = id_to_action(a)
        next_state, r, done, info = env.step(a)
        next_state = rgb2gray(next_state)
        episode_reward += r
        state = np.append(state[1:], [next_state], axis=0)
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to train the network.')
    parser.add_argument('-m', '--model', dest='model', type=str,
                        help='Path to saved model.')
    parser.add_argument('-p', '--past', dest='history_length', type=int,
                        help='Number of frames to look back to create one data input.')
    parser.add_argument('-t', '--timesteps', dest='max_timesteps', type=int,
                        help='Number of yimesteps to run an episode for.')

    args = parser.parse_args()
    if args.model is None:
        print("Enter model path!")
        sys.exit(1)
    if args.history_length is None:
        print("Enter history length!")
        sys.exit(1)
    model_path = args.model
    history_length = args.history_length
    max_timesteps = args.max_timesteps

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    # history_length = 20
    # lr = 0.001
    agent = BCAgent(history_length=history_length)
    agent.load(model_path)

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering,
                                    history_length=history_length,
                                    max_timesteps=max_timesteps)
        episode_rewards.append(episode_reward)
        print(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
