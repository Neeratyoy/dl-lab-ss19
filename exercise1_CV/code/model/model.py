import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, model_urls

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResNetConv(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        intermediate = []
        x = self.layer1(x); intermediate.append(x)
        x = self.layer2(x); intermediate.append(x)
        x = self.layer3(x); intermediate.append(x)

        return x, intermediate


class ResNetModel(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        # base network
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])

        # other network modules
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 34)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        if pretrained:
            self.res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    def forward(self, inputs, filename=''):
        x, _ = self.res_conv(inputs)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x + 0.5
        return x


def softmax(x):
    smax = torch.nn.Softmax(dim = 2)
    x_size = x.size()
    # Reshaping for spatial softmax
    x = x.view(x_size[0], x_size[1], x_size[2]* x_size[3])
    x = smax(x)
    return(x.view(*x_size))


class ResNetHourglass(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        # base network
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # removing output FC layer
        self.res_conv = nn.Sequential(*list(self.res_conv.children())[:-3])
        # outputs 256x64x64

        # creating transpose convolution layers
        # self.deconv1 = nn.Sequential(
        #                     nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,
        #                                        stride=2, padding=1), # output_size = 16
        #                     nn.BatchNorm2d(num_features=256),
        #                     nn.ReLU()
        #                 )
        self.deconv2 = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,
                                               stride=2, padding=1),  # output_size = 32
                            nn.BatchNorm2d(num_features=128),
                            nn.ReLU()
                        )
        self.deconv3 = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
                                               stride=2, padding=1),  # output_size = 64
                            nn.BatchNorm2d(num_features=64),
                            nn.ReLU()
                        )
        self.deconv4 = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4,
                                               stride=2, padding=1),  # output_size = 128
                            nn.BatchNorm2d(num_features=32),
                            nn.ReLU()
                        )
        self.deconv5 = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=32, out_channels=17, kernel_size=4,
                                               stride=2, padding=1),  # output_size = 256x256x17
                            nn.BatchNorm2d(num_features=17),
                            nn.ReLU()
                        )

    def forward(self, inputs, filename=''):
        x = self.res_conv(inputs)
        # x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        # computing soft-argmax and reshaping to a batch x 34 dimensional vector
        x = softmax(x)
        img_size = x.shape[-1]
        Wy = torch.arange(1, img_size+1).double().clone().expand(img_size, img_size) / img_size
        Wx = Wy.clone().transpose(0, 1)
        Wx = Wx.unsqueeze(0).expand(*x.shape).double().to(device)
        Wy = Wy.unsqueeze(0).expand(*x.shape).double().to(device)
        # print(x.is_cuda, Wx.is_cuda, Wy.is_cuda)
        result_x = torch.sum(x.double() * Wx, dim=(2,3))
        result_y = torch.sum(x.double() * Wy, dim=(2,3))
        result = torch.cat([torch.unsqueeze(result_x, 2), torch.unsqueeze(result_y, 2)], dim=2)
        x = result.view((result.shape[0], result.shape[1] * result.shape[2]))

        return x


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
        return x


class SegNet(nn.Module):
    def __init__(self, pretrained, task=1):
        super().__init__()
        self.task = task
        # base network
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # removing output FC layer
        # self.res_conv = nn.Sequential(*list(self.res_conv.children())[:-2])   # outputs 256x16x16
        self.res_conv = nn.ModuleList([*list(self.res_conv.children())[:-2]])

        if task == 1:
            self.upsample = nn.Sequential(
                                Interpolate(size=(256, 256), mode='bilinear'),
                                # 1x1 convoluton to downsample channels
                                nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1,
                                          stride=1, padding=0),
                                nn.Sigmoid()
                            )
        elif task == 2:
            self.deconv1 = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,
                                                   stride=2, padding=1),  # output_size = 16
                                nn.BatchNorm2d(num_features=256),
                                nn.ReLU()
                            )
            self.deconv2 = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,
                                                   stride=2, padding=1),  # output_size = 32
                                nn.BatchNorm2d(num_features=128),
                                nn.ReLU()
                            )
            self.deconv3 = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
                                                   stride=2, padding=1),  # output_size = 64
                                nn.BatchNorm2d(num_features=64),
                                nn.ReLU()
                            )
            self.deconv4 = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=8,
                                                   stride=4, padding=2),  # output_size = 256
                                                   # 1x1 convoluton to downsample channels
                                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1,
                                         stride=1, padding=0),
                                nn.Sigmoid()
                            )
        else:
            self.deconv1 = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,
                                                   stride=2, padding=1),  # output_size = 16
                                nn.BatchNorm2d(num_features=256),
                                nn.ReLU()
                            )
            # in_channels are multiplied by 2 - concatenation from skip connections
            self.deconv2 = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=256*2, out_channels=128, kernel_size=4,
                                                   stride=2, padding=1),  # output_size = 32
                                nn.BatchNorm2d(num_features=128),
                                nn.ReLU()
                            )
            self.deconv3 = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=128*2, out_channels=64, kernel_size=4,
                                                   stride=2, padding=1),  # output_size = 64
                                nn.BatchNorm2d(num_features=64),
                                nn.ReLU()
                            )
            self.deconv4 = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=64*2, out_channels=32, kernel_size=8,
                                                   stride=4, padding=2),  # output_size = 256
                                                   # 1x1 convoluton to downsample channels
                                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1,
                                         stride=1, padding=0),
                                nn.Sigmoid()
                            )


    def forward(self, inputs, filename=''):
        intermediate = []
        x = self.res_conv[0](inputs)
        for i in range(1, len(self.res_conv)-4):
            x = self.res_conv[i](x)
        for i in range(len(self.res_conv)-4, len(self.res_conv)-1):
            x = self.res_conv[i](x);
            intermediate.append(x)
        x = self.res_conv[-1](x)     # The bottleneck of volume 512 x 8 x 8

        # intermediate[0] -> 64 x 64 x 64
        # intermediate[1] -> 128 x 32 x 32
        # intermediate[2] -> 256 x 16 x 16

        if self.task == 1:
            x = self.upsample(x)
        elif self.task == 2:
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = self.deconv4(x)
        else: # task = 3 (skip connections)
            # intermediate[0] -> 64 x 64 x 64
            # intermediate[1] -> 128 x 32 x 32
            # intermediate[2] -> 256 x 16 x 16
            x = self.deconv1(x)
            # concatenating encoder output along channels
            x = torch.cat((intermediate[2], x), 1)
            x = self.deconv2(x)
            x = torch.cat((intermediate[1], x), 1)
            x = self.deconv3(x)
            x = torch.cat((intermediate[0], x), 1)
            x = self.deconv4(x)

        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# model = models.resnet18(pretrained=False)
# model.fc = Identity()
