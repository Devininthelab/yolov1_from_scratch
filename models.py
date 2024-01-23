import torch
import torch.nn as nn

"""
Information about the architectural configuration:
A Tuple is structured as (kernel_size, number of filters, stride, padding).
"M" simply represents max-pooling with a 2x2 pool size and 2x2 kernel.
The list is structured according to the data blocks, and ends with an integer representing the number of repetitions.
"""

# Describing convolutional and max-pooling layers, as well as the number of repetitions for convolutional blocks.
architecture_config = [
    (7, 64, 2, 3),  # Convolutional block 1
    "M",            # Max-pooling layer 1
    (3, 192, 1, 1), # Convolutional block 2
    "M",            # Max-pooling layer 2
    (1, 128, 1, 0), # Convolutional block 3
    (3, 256, 1, 1), # Convolutional block 4
    (1, 256, 1, 0), # Convolutional block 5
    (3, 512, 1, 1), # Convolutional block 6
    "M",            # Max-pooling layer 3
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # Convolutional block 7 (repeated 4 times)
    (1, 512, 1, 0), # Convolutional block 8
    (3, 1024, 1, 1),# Convolutional block 9
    "M",            # Max-pooling layer 4
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],  # Convolutional block 10 (repeated 2 times)
    (3, 1024, 1, 1),# Convolutional block 11
    (3, 1024, 2, 1),# Convolutional block 12
    (3, 1024, 1, 1),# Convolutional block 13
    (3, 1024, 1, 1),# Convolutional block 14
]



#Models

# A convolutional block is defined with Conv2d, BatchNorm2d, LeakyReLU
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            **kwargs)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        return self.fcs(x)
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for layer in architecture:
            if type(layer) == tuple:
                layers += [
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=layer[1],
                        kernel_size=layer[0],
                        stride=layer[2],
                        padding=layer[3]
                    )
                ]
                in_channels=layer[1]
            elif type(layer) == str:
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ]
            elif type(layer) == list:
                conv1 = layer[0]
                conv2 = layer[1]
                num_repeats = layer[2]
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels=in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    ]
                    layers += [
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]
                    in_channels = conv2[1]
            else:
                raise ValueError ("Config must be list, tuple, str")

        return nn.Sequential(*layers)

    def create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024 * S * S, 
                      out_features=496), #original paper this is 4096
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))# S, S, 30
        )

