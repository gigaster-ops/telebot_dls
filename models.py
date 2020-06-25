import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import torchvision.utils as vutils
import gc
from deepmux import create_model
from deepmux import get_model
from config import TOKEN_DEEPMUX
from copy import deepcopy as dc


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Model_monet():
    def __init__(self):
        print('Model Monet loading...')
        self.gen_ba = Generator()
        self.gen_ba.eval()
        self.size = 524
        self.gen_ba.load_state_dict(torch.load('pretrained_model/gen_ba80.pth', map_location='cpu'))

        '''self.gen_ba = create_model(
            self.gen_ba,
            'generator_v2',
            [1, 3, 524, 524],
            [1, 3, 524, 524],
            TOKEN_DEEPMUX
        )'''

        self.gen_ba = get_model(
            'generator_v2',
            TOKEN_DEEPMUX
        )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        print('Model Monet load!!!')

    async def predict(self, x):
        img = plt.imread(x + '.jpg')
        gc.collect()
        sh = img.shape
        tr = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((sh[0], sh[1])),
            transforms.ToTensor()
        ])
        img = self.transform(img)[None, :]
        out = self.gen_ba.run(img.detach().cpu().numpy())[0]
        out = np.transpose(tr(vutils.make_grid(torch.FloatTensor(dc(out)), normalize=True)).cpu().numpy(), (1, 2, 0))
        plt.imsave(x + '_pre.jpg', out)

        del img
        del out
        gc.collect()
