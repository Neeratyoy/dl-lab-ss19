import matplotlib.pyplot as plt
import numpy as np
import torch

from model.data_seg import get_data_loader
from utils.plot_util import plot_keypoints
from model.model import SegNet


if __name__ == '__main__':
    """
        Script to show samples of the dataset
    """
    # PATH_TO_CKPT = './trained_net.model'
    PATH_TO_CKPT = '/media/neeratyoy/Mars/Freiburg/SummerSemester19/DL_Lab/dl-lab-ss19/exercise1_CV/code/model_store/task_3/1/e_5.pt'

    # create device and model
    cuda = torch.device('cuda')
    model = SegNet(pretrained=True)
    model.load_state_dict(torch.load(PATH_TO_CKPT))
    model.to(cuda)
    #
    reader = get_data_loader(batch_size=1,
                                 is_train=False)

    for idx, (img, msk) in enumerate(reader):
        print('img', type(img), img.shape)
        print('msk', type(msk), msk.shape)
        pred = model(img.to(cuda), '').detach().cpu()[0]
        print('prd', type(pred), pred.shape, '\n')

        # turn image tensor into numpy array containing correctly scaled RGB image
        img_rgb = ((np.array(img) + 1.0)*127.5).round().astype(np.uint8).transpose([0, 2, 3, 1])

        # show
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(img_rgb[0]); #ax1.axis('off'); ax1.set_title("Image")
        ax1.imshow(msk[0], alpha=0.4); ax1.axis('off'); ax1.set_title("Ground Truth")
        ax2.imshow(img_rgb[0])
        ax2.imshow(torch.round(pred[0]), alpha=0.4)
        ax2.axis('off'); ax2.set_title("Prediction")
        plt.show()
