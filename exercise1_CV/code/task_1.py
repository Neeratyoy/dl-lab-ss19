import argparse
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from model.model import ResNetModel, ResNetHourglass
from model.data import get_data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_learning_curve(train, out_dir, name, test=None):
    '''
    Plots the learning curve - loss over epochs
    :param train: The set of training losses over epochs
    :param test: The set of test losses over epochs
    :param out_dir: Directory to save the plots
    :param name: Name appended to the plot saved 'learning_curve_[name].png'
    :return: void
    '''
    plt.clf()
    x = np.arange(1, len(train)*2+1, step=2)
    plt.plot(x, train, color='red', label='Training Loss')
    if test is not None:
        plt.plot(x, test, color='green', label='Test Set Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MPJPE')
    plt.xticks(x)
    plt.xlim(0,x[-1]+1)
    plt.legend()
    plt.grid(which='major', linestyle=':') #, axis='y')
    plt.grid(which='minor', linestyle='--', axis='y')
    plt.savefig(out_dir+'learning_curve_'+str(name)+'.png',dpi=300)


def normalize_keypoints(keypoints, img_shape):
    if img_shape[-1] != img_shape[-2]:
        raise ValueError("Only square images are supported")
    return keypoints/img_shape[-1]


# x.contiguous().view(34,2)


def mpjpe_eval(net, data_loader):
    pdist = nn.PairwiseDistance(p=2)
    metric = []
    with torch.no_grad():  # no gradient needed
        for i, data in enumerate(data_loader):
            # get the inputs
            inputs, labels, weights = data
            labels = normalize_keypoints(labels, inputs.shape)

            # reshaping weights to account for missing keypoint
            weights = np.repeat(weights.numpy(), 2)
            b_size = int(len(weights) / 34)
            weights = torch.from_numpy(weights.reshape((b_size, 34)))
            labels = weights.double() * labels.double()

            torch.cuda.empty_cache()
            # putting on GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            # forward + backward + optimize
            outputs = net(inputs, '')
            # ignore predictions for missing keypoints
            outputs = weights.double() * outputs.double()

            # MPJPE evaluation
            labels = labels.view(labels.shape[0], int(labels.shape[1] / 2), 2)
            labels = labels.view(labels.shape[0]*labels.shape[1], labels.shape[2])
            outputs = outputs.view(outputs.shape[0], int(outputs.shape[1] / 2), 2)
            outputs = outputs.view(outputs.shape[0]*outputs.shape[1], outputs.shape[2])
            metric.append(torch.mean(torch.sum(pdist(labels, outputs).view(b_size,17), 1) /
                                     (torch.sum(weights.double(), 1)/2)).item())
            # metric.append(torch.mean(dist).item())
    metric = np.mean(metric)
    return metric * inputs.shape[-1]


# gt = gt.view(gt.shape[0], int(gt.shape[1] / 2), 2)
# gt = gt.view(gt.shape[0]*gt.shape[1], gt.shape[2])
# preds = preds.view(preds.shape[0], int(preds.shape[1] / 2), 2)
# preds = preds.view(preds.shape[0]*preds.shape[1], preds.shape[2])
# torch.mean(torch.sum(pdist(gt, preds).view(2,17), 1) / (torch.sum(weights.double(), 1)/2)).item() * 256


def single_pass(net, data_loader, loss_criterion, optimizer, epoch_num,
                freq_log, **kwargs):
    running_loss = []
    for i, data in enumerate(data_loader):
        print(i)
        # get the inputs
        inputs, labels, weights = data
        labels = normalize_keypoints(labels, inputs.shape)

        # reshaping weights to account for missing keypoint
        weights = np.repeat(weights.numpy(), 2)
        b_size = int(len(weights) / 34)
        weights = torch.from_numpy(weights.reshape((b_size, 34)))
        labels = weights.double() * labels.double()

        torch.cuda.empty_cache()
        # putting on GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        weights = weights.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs, '')
        # ignore predictions for missing keypoints
        outputs = weights.double() * outputs.double()
        loss = loss_criterion(outputs, labels)
        # if backprop:
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        # print statistics
        running_loss.append(loss.item())
        if i % freq_log == freq_log-1:    # print every freq_log mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch_num, i + 1, np.mean(running_loss)))
        if (i+1) % 100 == 0:
            return running_loss
    return running_loss


def train(net, **kwargs):
    batch_size = kwargs['batch_size']
    valid = kwargs['valid']
    epochs = kwargs['epochs']
    freq_log = kwargs['freq_log']
    train_loader = get_data_loader(batch_size=batch_size,
                                   is_train=False)
    # test set
    if valid: val_loader = get_data_loader(batch_size=batch_size,
                                           is_train=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    train_mpjpe = []
    test_mpjpe = []
    for epoch in range(1, epochs+1):
        print("Epoch %s/%s" % (epoch, epochs))
        _ = single_pass(net, train_loader, criterion, optimizer, epoch, freq_log)

        # evaluating MPJPE every odd epoch
        if epoch % 2 == 1:
            train_mpjpe.append(mpjpe_eval(net, train_loader))
            print('-'*75)
            if valid:
                test_mpjpe.append(mpjpe_eval(net, val_loader))
                plot_learning_curve(train_mpjpe, "model_store/", "train_test", test=test_mpjpe)
                print("Epoch #%s: Training_MPJPE = %s px, Testing_MPJPE = %s px" %
                            (epoch, train_mpjpe[-1], test_mpjpe[-1]))
            else:
                plot_learning_curve(train_mpjpe, "model_store/", "train", test=None)
                print("Epoch #%s: Training_MPJPE = %s px" % (epoch, train_mpjpe[-1]))
            print('-'*75)

        # save model every 5 epochs, and first and last epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print("Taking model snapshot...")
            torch.save(net.state_dict(), "model_store/e_%s.pt" % epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to train the network.')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=128,
                        help='Integer giving the batch size.')
    parser.add_argument('-v', '--validate', dest='valid', type=bool, default=True,
                        choices=[True, False], help='Whether to validate after training.')
    parser.add_argument('-f', '--frequency_logging', dest='freq_log', type=int, default=200,
                        help='Number of batches after which logs and plots will be generated.')
    parser.add_argument('-t', '--task', dest='task', type=int, default=1,
                        choices=[1, 2], help='The task number to run for HPE.')
    parser.add_argument('-p', '--pretrained', dest='pretrained', type=bool, default=True,
                        choices=[True, False], help='To use pretrained ImageNet weights or not.')

    args = parser.parse_args()
    if args.task == 1:
        net = ResNetModel(pretrained=args.pretrained)
    else:
        net = ResNetHourglass(pretrained=args.pretrained)
    print(net)
    if torch.cuda.is_available():
        net.cuda()
    train(net, epochs=args.epochs, batch_size=args.batch_size, valid=args.valid,
          freq_log=args.freq_log)
