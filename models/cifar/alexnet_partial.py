'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
from split_conv import Conv2D_partial


__all__ = ['alexnet_partial']


class AlexNet_partial(nn.Module):

    def __init__(self, num_classes=10,part=1.0,zero_fixed_part=False):
        super(AlexNet_partial, self).__init__()
        self.features = nn.Sequential(
            Conv2D_partial(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),part,zero_fixed_part),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2D_partial(nn.Conv2d(64, 192, kernel_size=5, padding=2),part,zero_fixed_part),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2D_partial(nn.Conv2d(192, 384, kernel_size=3, padding=1),part,zero_fixed_part),
            nn.ReLU(inplace=True),
            Conv2D_partial(nn.Conv2d(384, 256, kernel_size=3, padding=1),part,zero_fixed_part),
            nn.ReLU(inplace=True),
            Conv2D_partial(nn.Conv2d(256, 256, kernel_size=3, padding=1),part,zero_fixed_part),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet_partial(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet_partial(**kwargs)
    return model
