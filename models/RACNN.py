import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import se_resnet


class RACNN(nn.Module):
    def __init__(self, num_classes):
        super(RACNN, self).__init__()

        self.backbone = se_resnet(num_classes)

        self.feature_pool = nn.AdaptiveMaxPool2d(1)

        self.apn = nn.Sequential(
            nn.Linear(512 * 5 * 5, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )

        self.crop_resize = AttentionCropLayer()

        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)

    def forward(self, x):

        feature1 = self.backbone(x)
        pool1 = self.feature_pool(feature1)
        atten1 = self.apn(feature1.view(-1, 512 * 5 * 5))
        scaledA_x = self.crop_resize(x, atten1 * 80)

        feature2 = self.backbone(scaledA_x)
        pool2 = self.feature_pool(feature2)


        pool1 = pool1.view(-1, 512)
        pool2 = pool2.view(-1, 512)


        """#Feature fusion
        scale123 = torch.cat([pool5, pool5_A, pool5_AA], 1)
        scale12 = torch.cat([pool5, pool5_A], 1)
        """

        logits1 = self.classifier1(pool1)
        logits2 = self.classifier2(pool2)

        return [logits1, logits2]#, [feature1, feature2], atten1


class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        h = lambda x: 1 / (1 + torch.exp(-10 * x))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size)
        x = torch.stack([unit] * 3)
        y = torch.stack([unit.t()] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()

        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]
            tx = tx if tx > (in_size / 3) else in_size / 3
            tx = tx if (in_size / 3 * 2) > tx else (in_size / 3 * 2)
            ty = ty if ty > (in_size / 3) else in_size / 3
            ty = ty if (in_size / 3 * 2) > ty else (in_size / 3 * 2)
            tl = tl if tl > (in_size / 3) else in_size / 3

            w_off = int(tx - tl) if (tx - tl) > 0 else 0
            h_off = int(ty - tl) if (ty - tl) > 0 else 0
            w_end = int(tx + tl) if (tx + tl) < in_size else in_size
            h_end = int(ty + tl) if (ty + tl) < in_size else in_size

            mk = (h(x - w_off) - h(x - w_end)) * (h(y - h_off) - h(y - h_end))
            xatt = images[i] * mk

            xatt_cropped = xatt[:, h_off: h_end, w_off: w_end]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            xamp = F.upsample(before_upsample, size=(80, 80), mode='bilinear', align_corners=True)
            ret.append(xamp.data.squeeze())

        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor = self.saved_variables[0], self.saved_variables[1]
        in_size = 80
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)

        #         show_image(inputs.cpu().data[0])
        #         show_image(ret_tensor.cpu().data[0])
        #         plt.imshow(norm[0].cpu().numpy(), cmap='gray')

        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size / 3 * 2)
        short_size = (in_size / 3)
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x < short_size) + (x >= long_size) + (y < short_size) + (y >= long_size)) > 0).float() * 2 - 1

        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))

        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()

        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret


class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """

    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)

