import torch.nn as nn
import torchvision.models as models


class ConvModel(nn.Module):
    """Base model with convolutional layers (Resnet18)

    """

    def __init__(self, out_dim, dataset: str = ''):
        super(ConvModel, self).__init__()
        self.resnet_model = models.resnet18(pretrained=False, num_classes=out_dim)

        self.backbone = self.resnet_model
        if dataset.upper() == 'MNIST':
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.backbone(x)