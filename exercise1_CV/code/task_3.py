import argparse
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from model.model import SegNet
from model.data_seg import get_data_loader

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
    plt.ylabel('IoU')
    plt.xticks(x)
    plt.xlim(0,x[-1]+1)
    plt.legend()
    plt.grid(which='major', linestyle=':') #, axis='y')
    plt.grid(which='minor', linestyle='--', axis='y')
    plt.savefig(out_dir+'learning_curve_'+str(name)+'.png',dpi=300)


# x.contiguous().view(34,2)

def iou(batch1, batch2):
    batch_size, _, H, W = batch1.shape
    intersection = batch1.view(batch_size, H * W) * batch2.view(batch_size, H * W)
    union = batch1.view(batch_size, H * W) + batch2.view(batch_size, H * W) - intersection
    return(torch.mean(torch.sum(intersection, dim=1) / torch.sum(union, dim=1)).item())


def iou_eval(net, data_loader):
    pdist = nn.PairwiseDistance(p=2)
    metric = []
    with torch.no_grad():  # no gradient needed
        for i, data in enumerate(data_loader):
            # print(i)
            # get the inputs
            imgs, masks = data

            torch.cuda.empty_cache()
            # putting on GPU
            imgs = imgs.to(device)
            masks = masks.to(device)

            # forward + backward + optimize
            outputs = net(imgs, '')

            # iou evaluation
            metric.append(iou(outputs, masks))

    metric = np.mean(metric)
    return metric


# IoU calculations
# torch.sum(mask * outputs, dim=(1,2)) / torch.sum(mask + outputs - mask * outputs, dim=(1,2))
# Softamx on output
# softmax = torch.nn.Softmax()
# outputs = softmax(outputs)
#
# loss = torch.nn.BCELoss()
# loss(outputs, mask.long())


def single_pass(net, data_loader, loss_criterion, optimizer, epoch_num,
                freq_log, **kwargs):
    running_loss = []
    for i, data in enumerate(data_loader):
        print(i)
        # get the inputs
        imgs, masks = data

        torch.cuda.empty_cache()
        # putting on GPU
        imgs = imgs.to(device)
        masks = masks.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(imgs, '')

        loss = loss_criterion(outputs.view(outputs.shape[0], outputs.shape[2],
                                           outputs.shape[3]), masks)
        # if backprop:
        loss.backward()
        optimizer.step()

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

    criterion = nn.BCELoss()  # binary cross entropy loss
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    train_IoU = []
    test_IoU = []
    for epoch in range(1, epochs+1):
        print("Epoch %s/%s" % (epoch, epochs))
        _ = single_pass(net, train_loader, criterion, optimizer, epoch, freq_log)

        # evaluating IoU every odd epoch
        if epoch % 2 == 1:
            train_IoU.append(iou_eval(net, train_loader))
            print('-'*75)
            if valid:
                test_IoU.append(iou_eval(net, val_loader))
                plot_learning_curve(train_IoU, "model_store/task_1/", "train_test", test=test_IoU)
                print("Epoch #%s: Training_IoU = %s px, Testing_IoU = %s px" %
                            (epoch, train_IoU[-1], test_IoU[-1]))
            else:
                plot_learning_curve(train_IoU, "model_store/task_1/", "train", test=None)
                print("Epoch #%s: Training_IoU = %s px" % (epoch, train_IoU[-1]))
            print('-'*75)

        # save model every 5 epochs, and first and last epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print("Taking model snapshot...")
            torch.save(net.state_dict(), "model_store/task_1/e_%s.pt" % epoch)


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
                        choices=[1, 2, 3], help='The task number to run for Segmentation.')
    parser.add_argument('-p', '--pretrained', dest='pretrained', type=bool, default=True,
                        choices=[True, False], help='To use pretrained ImageNet weights or not.')

    args = parser.parse_args()
    # if args.task == 1:
    #     net = ResNetModel(pretrained=args.pretrained)
    # else:
    #     net = ResNetHourglass(pretrained=args.pretrained)
    net = SegNet(pretrained=args.pretrained)
    print(net)
    if torch.cuda.is_available():
        net.cuda()
    train(net, epochs=args.epochs, batch_size=args.batch_size, valid=args.valid,
          freq_log=args.freq_log)
