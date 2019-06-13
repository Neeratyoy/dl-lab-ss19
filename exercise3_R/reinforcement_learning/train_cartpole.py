import sys
sys.path.append("../")

import numpy as np
import gym
import itertools as it
from agent.dqn_agent import DQNAgent, device
from tensorboard_evaluation import *
from agent.networks import MLP
from utils import EpisodeStats
import time
import math
import argparse


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000, eps=0.1):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:
        step += 1
        action_id = agent.act(state=state, deterministic=deterministic, eps=eps)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()
            time.sleep(0.1)

        if terminal or step > max_timesteps:
            break

    print(step)
    return stats

def train_online(env, agent, num_episodes, decay_steps, rendering, max_timesteps,
                 num_eval_episodes, eval_cycle, model_dir="./models_cartpole",
                 tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard = Evaluation(store_dir=os.path.join(tensorboard_dir, "cartpole"), name="cartpole",
                             stats=["episode_reward", "a_0", "a_1", "eval_reward"])

    # Training
    eval_score = 0
    for i in range(num_episodes):
        # Linear epsilon decay
        if decay_steps is not 0:
            start = 1; end = agent.epsilon
            delta = (start - end) / decay_steps
            eps = max(start - delta * i, agent.epsilon)
        else:
            eps = agent.epsilon
        stats = run_episode(env, agent, deterministic=False, do_training=True, eps=eps,
                            max_timesteps=max_timesteps)

        print("episode: ",i, '; reward:', stats.episode_reward)
        # Evaluation
        if i % eval_cycle == 0:
            r = []
            for j in range(num_eval_episodes):
                eval_stats = run_episode(env, agent, deterministic=True, do_training=False,
                                         rendering=rendering, max_timesteps=max_timesteps)
                r.append(eval_stats.episode_reward)
            print("Evaluation score: {}\n".format(r))
            eval_score = np.mean(r)

        tensorboard.write_episode_data(i, eval_dict={  "episode_reward" : stats.episode_reward,
                                                       "a_0" : stats.get_action_usage(0),
                                                       "a_1" : stats.get_action_usage(1),
                                                       "eval_reward" : eval_score})

        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))
    tensorboard.close_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to learn cartpole.')
    parser.add_argument('-n', '--num_episodes', dest='num_episodes', type=int, default=200,
                        help='Number of episodes to run.')
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.1,
                        help='The epsilon used for e-greedy policy.')
    parser.add_argument('-d', '--decay', dest='decay_steps', type=int, default=50,
                        help='Number of episodes for which epsilon is decayed for exploration \
                              till the chosen e-greedy policy. To disable linear decay, set as 0.')
    parser.add_argument('-f', '--f_eval', dest='eval_cycle', type=int, default=20,
                        help='Frequency of evaluations.')
    parser.add_argument('-v', '--validate', dest='num_eval_episodes', type=int, default=5,
                        help='Number of episodes to evaluate.')
    parser.add_argument('-r', '--rendering', dest='rendering', type=str, default='True',
                        choices=['True', 'False'], help='Number of episodes to evaluate.')
    parser.add_argument('-t', '--timesteps', dest='max_timesteps', type=int, default=200,
                        help='Maximum timesteps for an episode.')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=64,
                        help='Batch size to sample from the experience replay.')
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-4,
                        help='Learning rate/Step size for the optimizer.')
    parser.add_argument('-g', '--gamma', dest='gamma', type=float, default=0.95,
                        help='The discount factor for computing target returns.')

    args = parser.parse_args()

    num_episodes = args.num_episodes
    eval_cycle = args.eval_cycle
    num_eval_episodes = args.num_eval_episodes
    decay_steps = args.decay_steps
    epsilon = args.epsilon
    rendering = True if args.rendering=='True' else False
    max_timesteps = args.max_timesteps
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    gamma = args.gamma

    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped
    state_dim = 4
    num_actions = 2

    Q = MLP(state_dim, num_actions)
    Q_target = MLP(state_dim, num_actions)

    agent = DQNAgent(Q, Q_target, num_actions, epsilon=epsilon, gamma=gamma,
                     lr=learning_rate, batch_size=batch_size)
    train_online(env, agent, num_episodes=num_episodes, decay_steps=decay_steps,
                 num_eval_episodes=num_eval_episodes, eval_cycle=eval_cycle,
                 rendering=rendering, max_timesteps=max_timesteps)
