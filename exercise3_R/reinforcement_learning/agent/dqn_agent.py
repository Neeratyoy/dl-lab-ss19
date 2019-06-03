import torch
import tensorflow as tf
import numpy as np
from agent.replay_buffer import ReplayBuffer

device = "cuda" if torch.cuda.is_available() else "cpu"

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=32, epsilon=0.1,
                 tau=0.01, lr=1e-4, capacity=1e5, history_length=0):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tao: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.to(device)
        self.Q_target = Q_target.to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(capacity)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions


    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        states, actions, next_states, rewards, dones = self.replay_buffer.next_batch(self.batch_size)
        ## Double Q-learning
        preds = self.Q(torch.tensor(states).to(device).float()).cpu()
        a = torch.tensor(actions) #torch.argmax(torch.tensor(actions), 1)
        index = torch.stack((a, torch.zeros(a.size()).long()), 1)
        q_base = torch.gather(preds, 1, index.long())[:,0]

        next_preds = self.Q(torch.tensor(next_states).to(device).float()).cpu()
        next_a = torch.argmax(next_preds, 1)
        index = torch.stack((next_a, torch.zeros(next_a.size()).long()), 1)
        next_target_preds =  self.Q_target(torch.tensor(next_states).to(device).float()).cpu()
        q_target = torch.gather(next_target_preds, 1, index.long())[:,0]
        # reward signal
        end = np.where(dones)
        q_target[end] = 0.0
        q_target = torch.tensor(rewards).float() + self.gamma * q_target

        del(states, actions, next_states, rewards, dones)

        ## DQN
        # preds = self.Q(torch.tensor(states).to(device).float()).cpu()
        # a = torch.tensor(actions) #torch.argmax(preds, 1)
        # index = torch.stack((a, torch.zeros(a.size()).long()), 1)
        # q_base = torch.gather(preds, 1, index.long())[:,0]
        #
        # next_preds = self.Q_target(torch.tensor(next_states).to(device).float()).cpu()
        # q_target = torch.tensor(rewards).float() + self.gamma * torch.max(next_preds, 1)[0]

        # Update Q base
        self.optimizer.zero_grad()
        loss = self.loss_function(q_base, q_target)
        loss.backward()
        self.optimizer.step()

        # Update Q target (Polyak averaging)
        ### tau = 0.01; tau * Q + (1 - tau) * Q_target
        for name, param in self.Q.named_parameters():
            new_weights = (1 - self.tau) * self.Q_target.state_dict()[name].data + \
                          self.tau * self.Q.state_dict()[name].data
            self.Q_target.state_dict()[name].data.copy_(new_weights)


    def act(self, state, deterministic, eps, task="cartpole"):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        with torch.no_grad():
            r = np.random.uniform()
            if deterministic or r > eps:
                # take greedy action (argmax)
                a = self.Q(torch.tensor(state).to(device).float().unsqueeze(0)).detach().cpu()
                action_id = torch.argmax(a).item()
            else:
                # sample random action
                # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
                # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
                # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
                if task == "cartpole":
                    action_id = np.random.choice([0, 1])
                else:
                    action_id = np.random.choice([0, 1, 2, 3, 4],
                                                 p = [0.35, 0.19, 0.19, 0.25, 0.02])
                                                 # p = [0.5, 0.14, 0.14, 0.2, 0.02])

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        # self.Q.load_state_dict(torch.load(file_name))
        # self.Q_target.load_state_dict(torch.load(file_name))
