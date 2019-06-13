# export DISPLAY=:0

import sys
sys.path.append("../")

import numpy as np
import gym
from agent.dqn_agent import DQNAgent
from agent.networks import CNN
from tensorboard_evaluation import *
import itertools as it
from utils import *
import argparse


def run_episode(env, agent, deterministic, skip_frames=0, do_training=True, rendering=False,
                max_timesteps=1000, history_length=0, eps=0.1):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    stats = EpisodeStats()

    # Save history
    image_hist = []


    step = 0
    state = env.reset()
    i = 15

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    state = state_preprocessing(state)
    while True:
        action_id = agent.act(state=state, deterministic=deterministic, eps=eps, task="carracing")
        action = id_to_action(action_id)
        reward = 0
        while i>0:
            # To enforce an action in the zooming in of the race track
            action = id_to_action(3)
            _, r, _, _ = env.step(action)
            reward += r
            i = i-1
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                 break

        next_state = state_preprocessing(next_state)
        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps :
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, decay_steps, rendering, max_timesteps,
                 num_eval_episodes, eval_cycle, skip_frames, full_timesteps,
                 model_dir="./models_carracing", tensorboard_dir="./tensorboard"):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "carracing"),
                             stats=["episode_reward", "straight", "left", "right", "accel", "brake", "avg_eval_reward"],
                             name='carracing')

    for i in range(num_episodes):
        # Max timesteps schedule
        ### A linear scehdule starting from min_timesteps to full_timesteps
        min_timesteps = 200
        max_ts = np.arange(start=min_timesteps, stop=max_timesteps,
                           step = (max_timesteps-min_timesteps)/full_timesteps)
        ### Constant max_timesteps after full_timesteps
        max_ts = np.append(max_ts, np.repeat(max_timesteps, num_episodes - len(max_ts)))[i]

        # Linear epsilon decay
        if decay_steps is not 0:
            start = 1; end = agent.epsilon
            delta = (start - end) / decay_steps
            eps = max(start - delta * i, agent.epsilon)
        else:
            eps = agent.epsilon


        stats = run_episode(env, agent, max_timesteps=max_ts, deterministic=False,
                            do_training=True, eps=eps, rendering=rendering,
                            skip_frames=skip_frames)

        print("episode: ",i, '; reward:', stats.episode_reward, '; max_ts: ', max_ts, '; len: ', len(agent.replay_buffer._data.states))
        r = None
        # Evaluation
        if i % eval_cycle == 0:
            r = []
            for j in range(num_eval_episodes):
                eval_stats = run_episode(env, agent, deterministic=True, do_training=False,
                                         rendering=rendering, max_timesteps=max_ts,
                                         skip_frames=skip_frames) #max_timesteps)
                r.append(eval_stats.episode_reward)
            print("Evaluation score: Mean={}; Std.={}; Max={}\n".format(np.mean(r), np.std(r), np.max(r)))
            r = np.mean(r)
        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward,
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE),
                                                      "avg_eval_reward": r
                                                    })
        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent_{}.pt".format(i)))
    tensorboard.close_session()


def state_preprocessing(state):
    return (rgb2gray(state).reshape(1, 96, 96) / 255.0)


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
    parser.add_argument('-s', '--skip_frames', dest='skip_frames', type=int, default=10,
                        help='Number of frames to skip.')
    parser.add_argument('-x', '--full_timesteps', dest='full_timesteps', type=int, default=None,
                        help='Maximum number of episodes by which timesteps per episode is equal to the \
                              max timesteps allowed per episode.')

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
    skip_frames = args.skip_frames
    full_timesteps = args.full_timesteps if args.full_timesteps is not None else num_episodes

    num_actions = 5

    env = gym.make('CarRacing-v0').unwrapped

    Q = CNN()
    Q_target = CNN()
    agent = DQNAgent(Q, Q_target, num_actions, epsilon=epsilon, gamma=gamma,
                     lr=learning_rate, batch_size=batch_size, capacity=10000)

    train_online(env, agent, num_episodes=num_episodes, decay_steps=decay_steps, rendering=rendering,
                 max_timesteps=max_timesteps, num_eval_episodes=num_eval_episodes,
                 eval_cycle = eval_cycle, skip_frames=skip_frames, full_timesteps=full_timesteps,
                 model_dir="./models_carracing")
