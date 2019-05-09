import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch

from model.data_seg import get_data_loader
from utils.plot_util import plot_keypoints
from model.model import SegNet


def iou(batch1, batch2):
    batch_size, _, H, W = batch1.shape
    intersection = batch1.view(batch_size, H * W) * batch2.view(batch_size, H * W)
    union = batch1.view(batch_size, H * W) + batch2.view(batch_size, H * W) - intersection
    return(torch.mean(torch.sum(intersection, dim=1) / torch.sum(union, dim=1)).item())


if __name__ == '__main__':
    """
        Script to show samples of the dataset
    """
    parser = argparse.ArgumentParser(description='Arguments to select the type of network.')
    parser.add_argument('-t', '--task', dest='task', type=int, default=1, choices=[1,2,3],
                        help='Select the type of segmentation network.')
    args = parser.parse_args()

    # PATH_TO_CKPT = './trained_net.model'
    PATH_TO_CKPT = 'model_store/task_3/1/task3_1_e_10.pt'
    # PATH_TO_CKPT2 = '/media/neeratyoy/Mars/Freiburg/SummerSemester19/DL_Lab/dl-lab-ss19/exercise1_CV/code/model_store/task_3/2/task3_2_e_15.pt'

    # create device and model
    cuda = torch.device('cuda')
    model = SegNet(pretrained=True, task=args.task)
    model.load_state_dict(torch.load(PATH_TO_CKPT))
    model.to(cuda)
    #
    reader = get_data_loader(batch_size=1, is_train=False)

    for idx, (img, msk) in enumerate(reader):
        if idx!=76:
            continue
        print(idx)
        print('img', type(img), img.shape)
        print('msk', type(msk), msk.shape)
        pred = model(img.to(cuda), '').detach().cpu()[0]
        print('prd', type(pred), pred.shape, iou(pred.unsqueeze(0), msk.unsqueeze(0)), '\n')

        # turn image tensor into numpy array containing correctly scaled RGB image
        img_rgb = ((np.array(img) + 1.0)*127.5).round().astype(np.uint8).transpose([0, 2, 3, 1])

        # show
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        # ax1 = fig.add_subplot(121)
        # ax2 = fig.add_subplot(122)
        # ax1.imshow(img_rgb[0]); ax1.axis('off'); ax1.set_title("Image")
        # ax1.imshow(msk[0], alpha=0.4); #ax1.axis('off'); ax1.set_title("Upsampling")
        ax2.imshow(img_rgb[0])
        ax2.imshow(torch.round(pred[0]), alpha=0.4)
        # # ax2.imshow(pred[0], alpha=0.4)
        ax2.axis('off'); ax2.set_title("Upsampling")
        plt.savefig('3_1.png', dpi=300)
        # plt.clf()
        # if (idx+1) % 100 == 0:
        #     break
        # plt.show()
