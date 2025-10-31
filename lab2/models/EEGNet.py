import torch.nn as nn
import torch.nn.functional as F
import torch


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding="same")
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# TODO implement EEGNet model


class EEGNet(nn.Module):
    def __init__(self, args):
        super(EEGNet, self).__init__()
        self.dropoutRate = args.dropout_rate
        self.activation = args.activation_function
        self.alpha = args.elu_alpha

        self.conv1 = nn.Conv2d(1, 16, (1, 51), padding=(
            0, 25), bias=False)  # (1, 64)
        # self.conv1 = nn.Conv2d(1, self.F1, (1, 64))  #(1, 64)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        # DepthwiseConv2D D * F1 (C, 1)
        '''When groups == in_channels and out_channels == K * in_channels, 
        where K is a positive integer, this operation is also known as a “depthwise convolution”.
        '''
        # 16, 1, 2, 1
        self.depthwise_conv1 = nn.Conv2d(in_channels=16, out_channels=32, stride=(1, 1),
                                         kernel_size=(2, 1), groups=16)
        self.batchnorm2 = nn.BatchNorm2d(32, False)
        self.avgpooling = nn.AvgPool2d(
            kernel_size=(1, 4), stride=(1, 4), padding=0)

        # Layer 2
        # self.sep_depthwise = nn.Conv2d(16, self.F2, kernel_size=(1, 16), groups=16, bias=False)
        self.separable_conv = nn.Conv2d(32, 32, kernel_size=(
            1, 15), stride=(1, 1), bias=False, padding=(0, 7))

        self.batchnorm3 = nn.BatchNorm2d(32, False)
        self.avgpooling2 = nn.AvgPool2d(
            kernel_size=(1, 8), stride=(1, 8), padding=0)

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        # self.fc1 = nn.Linear(384, 1)
        self.fc1 = nn.Linear(in_features=736, out_features=2, bias=True)

        if self.activation == 'relu':
            self.activation_func = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.activation_func = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.activation_func = nn.ELU(alpha=self.alpha)
        elif self.activation == 'selu':
            self.activation_func = nn.SELU()
        else:
            raise ValueError('Invalid activation function.')

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv1(x)
        x = self.batchnorm2(x)
        x = self.activation_func(x)
        x = self.avgpooling(x)
        x = F.dropout(x, self.dropoutRate)

        # Block 2
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = self.activation_func(x)

        x = self.avgpooling2(x)
        x = F.dropout(x, self.dropoutRate)

        # FC Layer
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)  # logits for CrossEntropyLoss
        return x


# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(self, args):
        super(DeepConvNet, self).__init__()
        self.dropout_rate = args.dropout_rate
        self.elu_alpha = args.elu_alpha
        self.activation = args.activation_function

        # Block 1
        self.block1_conv_time = nn.Conv2d(
            1, 25, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.block1_conv_spat = nn.Conv2d(
            25, 25, kernel_size=(2, 1), bias=False)
        self.block1_bn = nn.BatchNorm2d(25, eps=1e-5, momentum=0.1)
        self.block1_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Block 2
        self.block2_conv = nn.Conv2d(
            25, 50, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.block2_bn = nn.BatchNorm2d(50, eps=1e-5, momentum=0.1)
        self.block2_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Block 3
        self.block3_conv = nn.Conv2d(
            50, 100, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.block3_bn = nn.BatchNorm2d(100, eps=1e-5, momentum=0.1)
        self.block3_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Block 4
        self.block4_conv = nn.Conv2d(
            100, 200, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.block4_bn = nn.BatchNorm2d(200, eps=1e-5, momentum=0.1)
        self.block4_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.dropout = nn.Dropout(self.dropout_rate)

        if self.activation == 'relu':
            self.activation_func = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.activation_func = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.activation_func = nn.ELU(alpha=self.elu_alpha)
        elif self.activation == 'selu':
            self.activation_func = nn.SELU()
        else:
            raise ValueError('Invalid activation function.')

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(200, 2)

    def forward(self, x):
        # Block 1
        x = self.block1_conv_time(x)
        x = self.block1_conv_spat(x)
        x = self.block1_bn(x)
        x = self.activation_func(x)
        x = self.block1_pool(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        # Block 2
        x = self.block2_conv(x)
        x = self.block2_bn(x)
        x = self.activation_func(x)
        x = self.block2_pool(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        # Block 3
        x = self.block3_conv(x)
        x = self.block3_bn(x)
        x = self.activation_func(x)
        x = self.block3_pool(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        # Block 4
        x = self.block4_conv(x)
        x = self.block4_bn(x)
        x = self.activation_func(x)
        x = self.block4_pool(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        # Head
        x = self.global_pool(x)  # (N, 200, 1, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)   # logits for CrossEntropyLoss
        return x
