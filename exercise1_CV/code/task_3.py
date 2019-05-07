import json
import argparse
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from model.model import SegNet
from model.data_seg import get_data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parameter_count(model):
    return(sum(p.numel() for p in model.parameters() if p.requires_grad))


def model_size_in_MB(model):
    # 72 verified using sys.getsizeof
    return((parameter_count(model) * 72) / (1024*1024))


def load_n_plot_iou(file1, file2, name1, name2, out_dir):
    '''
    Plots the learning curve - loss over epochs
    :param train: The set of training losses over epochs
    :param test: The set of test losses over epochs
    :param out_dir: Directory to save the plots
    :param name: Name appended to the plot saved 'learning_curve_[name].png'
    :return: void
    '''
    data1 = []
    data2 = []
    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)
    try:
        data1 = data1['test']
        data2 = data2['test']
    except:
        print("Need a dict of lists with at least one key value as 'test'.")
        return
    size = min(len(data1), len(data2))
    data1 = data1[:size]
    data2 = data2[:size]
    plt.clf()
    x = np.arange(1, len(data1)*2+1, step=2)
    plt.plot(x, data1, color='red', label=name1)
    plt.plot(x, data2, color='green', label=name2)
    plt.title('Comparison of IoU on Test set')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.xticks(x)
    plt.xlim(0,x[-1]+1)
    plt.legend()
    plt.grid(which='major', linestyle=':') #, axis='y')
    plt.grid(which='minor', linestyle='--', axis='y')
    plt.savefig(out_dir+'iou_comparison.png',dpi=300)


def plot_iou(train, out_dir, name, test=None):
    '''
    Plots the learning curve - loss over epochs
    :param train: The set of training losses over epochs
    :param test: The set of test losses over epochs
    :param out_dir: Directory to save the plots
    :param name: Name appended to the plot saved 'iou_[name].png'
    :return: void
    '''
    plt.clf()
    x = np.arange(1, len(train)*2+1, step=2)
    plt.plot(x, train, color='red', label='Training IoU')
    if test is not None:
        plt.plot(x, test, color='green', label='Test IoU')
    plt.title('IoU over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.xticks(x)
    plt.xlim(0,x[-1]+1)
    plt.legend()
    plt.grid(which='major', linestyle=':') #, axis='y')
    plt.grid(which='minor', linestyle='--', axis='y')
    plt.savefig(out_dir+'iou_'+str(name)+'.png',dpi=300)
    plot_data = {'train': train, 'test': test}
    with open(out_dir+'plot_data.json', 'w') as f:
        json.dump(plot_data, f)


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

            # forward
            outputs = torch.round(net(imgs, ''))

            # iou evaluation
            metric.append(iou(outputs, masks))

    metric = np.mean(metric)
    return metric


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
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss.append(loss.item())
        if i % freq_log == freq_log-1:    # print every freq_log mini-batches
            print("Epoch #%d; Batch %d/%d; Loss: %f" %
                  (epoch_num, i+1, len(data_loader), np.mean(running_loss)))
        # if (i+1) % 100 == 0:
        #     return running_loss
    return running_loss


def train(net, **kwargs):
    batch_size = kwargs['batch_size']
    valid = kwargs['valid']
    epochs = kwargs['epochs']
    freq_log = kwargs['freq_log']
    out_dir = kwargs['out_dir']

    # train set
    train_loader = get_data_loader(batch_size=batch_size, is_train=True)
    # test set
    if valid: val_loader = get_data_loader(batch_size=batch_size, is_train=False)

    criterion = nn.BCELoss()  # binary cross entropy loss
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    train_IoU = []
    test_IoU = []
    for epoch in range(1, epochs+1):
        print("Epoch %s/%s" % (epoch, epochs))
        _ = single_pass(net, train_loader, criterion, optimizer, epoch, freq_log)

        # evaluating IoU every odd epoch
        if epoch % 2 == 1:
            print('-'*75)
            train_IoU.append(iou_eval(net, train_loader))
            if valid:
                test_IoU.append(iou_eval(net, val_loader))
                plot_iou(train_IoU, out_dir, "train_test", test=test_IoU)
                print("Epoch #%s: Training_IoU = %s px, Testing_IoU = %s px" %
                            (epoch, train_IoU[-1], test_IoU[-1]))
            else:
                plot_iou(train_IoU, out_dir, "train", test=None)
                print("Epoch #%s: Training_IoU = %s px" % (epoch, train_IoU[-1]))
            print('-'*75)

        # save model every 5 epochs, and first and last epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print("Taking model snapshot...")
            torch.save(net.state_dict(), out_dir+"e_%s.pt" % epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to train the network.')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=128,
                        help='Integer giving the batch size.')
    parser.add_argument('-v', '--validate', dest='valid', type=str, default='True',
                        choices=['True', 'False'], help='Whether to validate after training.')
    parser.add_argument('-f', '--frequency_logging', dest='freq_log', type=int, default=200,
                        help='Number of batches after which logs and plots will be generated.')
    parser.add_argument('-t', '--task', dest='task', type=int, default=1,
                        choices=[1, 2, 3], help='The task number to run for Segmentation. \
                        1) Only upsampling after encoder. 2) 4 layer decoder after encoder. \
                        3) 4 layer decoder with skip connections from encoder.')
    parser.add_argument('-o', '--out', dest='out_dir', type=str, default='',
                        help='Directory to save models and plots.')

    args = parser.parse_args()
    net = SegNet(pretrained=True, task=args.task)
    print(net)
    print('~+~'*15)
    print('# of model parameters: ', parameter_count(net))
    print('Size of model (in MB): ', model_size_in_MB(net))
    print('~+~'*15)
    if torch.cuda.is_available():
        net.cuda()
    train(net, epochs=args.epochs, batch_size=args.batch_size, valid=args.valid,
          freq_log=args.freq_log, out_dir=args.out_dir)
