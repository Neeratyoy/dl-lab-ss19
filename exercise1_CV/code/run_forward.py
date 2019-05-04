import torch
import numpy as np
import matplotlib.pyplot as plt

from model.model import ResNetModel, ResNetConv, ResNetHourglass
from model.data import get_data_loader
from utils.plot_util import plot_keypoints


def normalize_keypoints(keypoints, img_shape):
    if img_shape[-1] != img_shape[-2]:
        raise ValueError("Only square images are supported")
    return keypoints/img_shape[-1]


if __name__ == '__main__':
    # PATH_TO_CKPT = './trained_net.model'
    PATH_TO_CKPT = '/media/neeratyoy/Mars/Freiburg/SummerSemester19/DL_Lab/dl-lab-ss19/exercise1_CV/code/model_store/e_1.pt'

    # create device and model
    cuda = torch.device('cuda')
    model = ResNetHourglass(pretrained=True)
    model.load_state_dict(torch.load(PATH_TO_CKPT))
    model.to(cuda)

    val_loader = get_data_loader(batch_size=1,
                                 is_train=False)

    for idx, (img, keypoints, weights) in enumerate(val_loader):
        img = img.to(cuda)
        keypoints = keypoints.to(cuda)
        weights = weights.to(cuda)

        # normalize keypoints to [0, 1] range
        keypoints = normalize_keypoints(keypoints, img.shape)

        # apply model
        pred = model(img, '')

        # reshaping weights to account for missing keypoint
        weights2 = weights.to("cpu")
        weights2 = np.repeat(weights2.detach().numpy(), 2)
        b_size = int(len(weights2) / 34)
        weights2 = torch.from_numpy(weights2.reshape((b_size, 34)))
        weights2 = weights2.to(cuda)
        keypoints2 = weights2.double() * keypoints.double()
        pred2 = weights2.double() * pred.double()

        # MPJPE evaluation
        V = torch.sum(weights, dim=1)/2
        pdist = torch.nn.PairwiseDistance(p=2)
        dist = pdist(pred2, keypoints2).double() / V.double()
        # dist = torch.transpose(torch.transpose(dist.double(), dim0=0, dim1=1) / V.double(),
        #                       dim0=0, dim1=1)
        print(torch.mean(dist).item() * img.shape[-1], img.shape[-1])

        # show results
        img_np = np.transpose(img.cpu().detach().numpy(), [0, 2, 3, 1])
        img_np = np.round((img_np + 1.0) * 127.5).astype(np.uint8)
        kp_pred = pred.cpu().detach().numpy().reshape([-1, 17, 2])
        kp_gt = keypoints.cpu().detach().numpy().reshape([-1, 17, 2])
        vis = weights.cpu().detach().numpy().reshape([-1, 17])

        for bid in range(img_np.shape[0]):
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(img_np[bid]), ax1.axis('off'), ax1.set_title('input + gt')
            plot_keypoints(ax1, kp_gt[bid], vis[bid], img_size=img_np[bid].shape[:2], draw_limbs=True, draw_kp=True)
            ax2.imshow(img_np[bid]), ax2.axis('off'), ax2.set_title('input + pred')
            plot_keypoints(ax2, kp_pred[bid], vis[bid], img_size=img_np[bid].shape[:2], draw_limbs=True, draw_kp=True)
            plt.show()
