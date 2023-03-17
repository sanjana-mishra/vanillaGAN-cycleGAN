import torch
import torch.nn as nn
import torch.nn.functional as F



def conv_layer(in_channels, out_channels, kernel_size, stride = 2, padding = 1, batch_norm = True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def deconv_layer(in_channels, out_channels, kernel_size, stride = 2, padding = 1, batch_norm = True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv = conv_layer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, padding = 1, batch_norm = True)
    def forward(self, x):
        return x + self.conv(x)



class Discriminator(nn.Module):
    def __init__(self, opts):
        super(Discriminator, self).__init__()
        self.opts = opts

        #####TODO: Define the discriminator network#####
        self.layers = nn.Sequential(
            conv_layer(3, 32, 4, stride=2, padding=1, batch_norm=True),
            nn.LeakyReLU(),
            conv_layer(32, 64, 4, stride=2, padding=1, batch_norm=True),
            nn.LeakyReLU(),
            conv_layer(64, 128, 4, stride=2, padding=1, batch_norm=True),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, 4, stride=1, padding=0)
        )
        if self.opts.d_sigmoid:
            self.layers.append(nn.Sigmoid())


    def forward(self, x):
        #####TODO: Define the forward pass#####
        out = self.layers(x)

        return out

class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.opts = opts

        #####TODO: Define the generator network######

        self.layers = nn.Sequential(
            deconv_layer(100, 128, 4, stride=1, padding=0, batch_norm=True),
            nn.ReLU(),
            deconv_layer(128, 64, 4, stride=2, padding=1, batch_norm=True),
            nn.ReLU(),
            deconv_layer(64, 32, 4, stride=2, padding=1, batch_norm=True),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        #####TODO: Define the forward pass#####
        out = self.layers(x)
        return out

class CycleGenerator(nn.Module):
    def __init__(self, opts):
        super(CycleGenerator, self).__init__()
        self.opts = opts

        #####TODO: Define the cyclegan generator network######
        self.layers1 = nn.Sequential(
            conv_layer(3, 32, 4, stride=2, padding=1, batch_norm=True),
            nn.ReLU(),
            conv_layer(32, 64, 4, stride=2, padding=1, batch_norm=True),
            nn.ReLU()
        )
        self.layers2 = ResNetBlock(64)
        self.layers3 = nn.Sequential(
            deconv_layer(64, 32, 4, stride=2, padding=1, batch_norm=True),
            nn.ReLU(),
            deconv_layer(32, 3, 4, stride=2, padding=1, batch_norm=True),
            nn.Tanh()
        )

    def forward(self, x):
        #####TODO: Define the forward pass#####
        out = self.layers1(x)
        out = self.layers2(out)
        out = self.layers3(out)
        return out


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, pred, label):
        loss = self.mseloss(pred, label)
        return loss

class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super(CycleConsistencyLoss, self).__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, pred, label):
        loss = self.l1loss(pred, label)
        return loss