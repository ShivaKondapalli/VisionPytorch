import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import numpy as np
import json
import scipy.io
from collections import OrderedDict

# Shape of CNN input length four.

# rank four tensor --> B * C * H * W

#  B : batch_size,
#  C: color channels: 3 for RGB and 1 for greyscale --> for input layer, for each successive hidden layer after
# first convolutional layer, it is number of filters in the previous layer.
# Because number of filters give us the number of output channels from a layer.
# . H : Height, W: Width.

# So x[1, 2, 221, 3] ---> this will give us the pixel value of our image.

# So here, we took the 2nd image form our batch.
# picked the blue color channel. (0: red, 1: green, 2: blue)
# selected the 221st row in our image matrix, and the 4th value in that row.

# output channels of a convolutional network after they have been passed are called feature maps.
# when the next convolutional layers accepts the input, it will operate on a subset of  H * W
# defined by the filter size, but on all the feature maps, i.e. the volume.

# out channels from previous layer = 64
# filter size in the current layers = 7 * 7. total number of filters = 32

# The filters would then be of size 64 * 7 * 7  dimensional. Each 2-D 7*7 filter
# will act on 7*7 patches of all the 64 features maps from the previous layer simultaneous.
# there will a total of 32 such filters, which means the present layer will send 32 activation maps.
# to the next layer.


class MyConv(nn.Module):
    def __init__(self):
        super(MyConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=120)
        self.fc1_bn = nn.BatchNorm1d(120)

        self.dropout = nn.Dropout(0.2)

        self.out = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), kernel_size=2))

        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1_bn(self.fc1(x)))

        x = self.dropout(x)
        x = F.relu(self.out(x))

        return F.log_softmax(x, dim=1)


def main():
    net = MyConv()
    r = torch.rand([5, 1, 28, 28])
    print(r)
    print(r.size())
    s = net.forward(r)
    print(s)
    print(s.size())
    print(net)


if __name__ == '__main__':
    main()



torch.optim.Optimizer


