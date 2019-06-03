from __future__ import print_function

import sys
sys.path.append("../")

import argparse
import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import torch

from utils import *
from agent.bc_agent import BCAgent, device
# from bc_agent import BCAgent, device
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1, augment='True'):
    '''
    Stacks images based on history_length
    Assumes that the images are gray scale already
    '''
    if augment == 'True':
        print("Augmenting training set...")
        # Creating placeholders for pre-processed training data
        X_aug = np.zeros((2 * X_train.shape[0], X_train.shape[-2], X_train.shape[-1]))
        y_aug = np.zeros(2 * X_train.shape[0])
        ### Data Augmentation - mirror the data and append below
        # Copying data
        X_aug[:len(X_train)] = X_train[:len(X_train)].copy()
        y_aug[:len(y_train)] = y_train[:len(y_train)].copy()
        j = len(X_train)
        # Mirroring data
        for i in range(len(X_train)):
            X_aug[j] = mirror_state(X_train[i])
            y_aug[j] = flip_direction(y_train[i])
            j = j+1
        # Re-assigning and cleaning memory
        X_train = X_aug.copy(); y_train = y_aug.copy()
        del(X_aug, y_aug)

    print("Preprocessing training set...")
    train_size = len(X_train) - history_length + 1
    X_stacked = np.array(np.random.randint(0, 255, (train_size, history_length,
                                                    X_train.shape[-2], X_train.shape[-1])),
                        dtype=np.float32)
    y_stacked = np.array(np.random.randint(0, 255, train_size), dtype=np.float32)
    for i in range(len(X_train)-history_length + 1):
        print("%s/%s" % (i+1, len(X_train)-history_length+1), end='\r')
        X_stacked[i] = np.stack(X_train[i:i+history_length])
        y_stacked[i] = y_train[i+history_length-1].copy()
        # print(y_train[history_length-1])

    print("\nPreprocessing validation set...")
    # Creating placeholders for pre-processed validation data
    valid_size = len(X_valid) - history_length + 1
    X_valid_stacked = np.array(np.random.randint(0, 255, (valid_size, history_length,
                                                    X_valid.shape[-2], X_valid.shape[-1])),
                        dtype=np.float32)
    y_valid_stacked = np.array(np.random.randint(0, 255, valid_size), dtype=np.float32)
    for i in range(len(X_valid)-history_length + 1):
        print("%s/%s" % (i+1, len(X_valid)-history_length+1), end='\r')
        X_valid_stacked[i] = np.stack(X_valid[i:i+history_length])
        y_valid_stacked[i] = y_valid[i+history_length-1].copy()
    print('\n')
    return X_stacked, y_stacked, X_valid_stacked, y_valid_stacked


def sample_batch(y_train, batch_size):
    classes = np.unique(y_train)
    size = int(batch_size / len(classes))
    index = np.array([])
    for c in classes:
        index = np.append(index, np.random.choice(np.where(y_train == c)[0], size, replace=False))
    index = np.array(index, dtype=int)
    return index


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, optimizer,
                history_length, freq, model_dir="./models", tensorboard_dir="./tensorboard"):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    agent = BCAgent(lr=lr, history_length=history_length, optimizer=optimizer)

    tensorboard_eval = Evaluation(store_dir=tensorboard_dir, name='logger',
                                  stats=['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy'])

    for i in range(n_minibatches):
        print("Batch %d/%d" % (i+1, n_minibatches), end='\r')
        # samples = np.random.randint(low=0, high=len(X_train), size=batch_size)
        samples = sample_batch(y_train, batch_size)
        train_pred, train_loss = agent.update(X_train[samples], y_train[samples])
        _, train_pred = torch.max(train_pred.data, 1)
        train_pred = train_pred.detach().cpu().numpy()
        valid_pred, valid_loss = agent.predict(X_valid, y_valid)
        _, valid_pred = torch.max(valid_pred.data, 1)
        valid_pred = valid_pred.detach().cpu().numpy()
        # valid_loss = agent.loss_criterion(valid_pred, np.array(y_valid, dtype=np.int64)).item()
        train_acc = balanced_accuracy_score(y_train[samples], train_pred)
        valid_acc = balanced_accuracy_score(y_valid, valid_pred)
        eval_dict = {
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_accuracy': train_acc,
            'valid_accuracy': valid_acc
        }
        tensorboard_eval.write_episode_data(episode=i+1, eval_dict=eval_dict)

        if (i+1) % freq == 0:
            file_name = model_dir + "/agent_" +str(i+1)+".pt"
            agent.save(file_name)
            print("Model saved in file: %s"  % file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to train the network.')
    parser.add_argument('-m', '--minibatches', dest='n_minibatches', type=int, default=1000,
                        help='Number of minibatches to train.')
    parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=64,
                        help='Integer giving the batch size.')
    parser.add_argument('-p', '--past', dest='history_length', type=int, default=1,
                        help='Number of frames to look back to create one data input.')
    parser.add_argument('-f', '--frequency_logging', dest='freq_log', type=int, default=100,
                        help='Number of batches after which model will be saved.')
    parser.add_argument('-o', '--optimizer', dest='optimizer', type=str, default='adam',
                        choices=['adam', 'rms'], help='Choose the optimizer.')
    parser.add_argument('-l', '--learning_rate', dest='lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('-s', '--split', dest='frac', type=float, default=0.1,
                        help='Fraction of dataset left out for validation.')
    parser.add_argument('-a', '--augment', dest='augment', type=str, default='True',
                        choices=['True', 'False'], help='To augment the data or not.')

    args = parser.parse_args()

    history_length = args.history_length
    n_minibatches = args.n_minibatches
    batch_size = args.batch_size
    lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam
    else:
         optimizer = torch.optim.RMSprop
    frac = args.frac
    freq_log = args.freq_log
    augment = args.augment

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data", frac=frac)

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid,
                                                       history_length=history_length,
                                                       augment=augment)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=n_minibatches,
                batch_size=batch_size, optimizer=optimizer, lr=lr,
                history_length=history_length, freq=freq_log)
