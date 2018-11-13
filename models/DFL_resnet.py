import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision
from model import se_resnet


class DFL(nn.Module):
    def __init__(self, k=10, nclass=300):
        super(DFL, self).__init__()
        self.k = k
        self.nclass = nclass

        # k channels for one class, nclass is total classes, therefore k * nclass for conv6
        self.backbone = se_resnet(nclass)

        conv6 = torch.nn.Conv2d(256, k * nclass, kernel_size=1, stride=1, padding=0)
        #pool6 = torch.nn.MaxPool2d((5, 5), stride=(5, 5), return_indices=True)
        pool6 = torch.nn.AdaptiveMaxPool2d(1)

        # Feature extraction root
        #self.conv1_conv4 = conv1_conv4

        # G-Stream
        #self.conv5 = conv5
        self.cls5 = nn.Sequential(
            nn.Conv2d(512, nclass, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(nclass),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # P-Stream
        self.conv6 = conv6
        self.pool6 = pool6
        self.cls6 = nn.Sequential(
            nn.Conv2d(k * nclass, nclass, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Side-branch
        self.cross_channel_pool = nn.AvgPool1d(kernel_size=k, stride=k, padding=0)

    def forward(self, x):
        batchsize = x.size(0)

        _, inter4, x_g = self.backbone(x)

        # Stem: Feature extraction
        #inter4 = self.conv1_conv4(x)

        # G-stream
        #x_g = self.conv5(inter4)
        out1 = self.cls5(x_g)
        out1 = out1.view(batchsize, -1)

        # P-stream ,indices is for visualization
        x_p = self.conv6(inter4)
        x_p = self.pool6(x_p)
        inter6 = x_p
        out2 = self.cls6(x_p)
        out2 = out2.view(batchsize, -1)

        # Side-branch
        inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
        out3 = self.cross_channel_pool(inter6)
        out3 = out3.view(batchsize, -1)

        return out1, out2, out3