import torch
import torch.nn as nn

class Mobilenet(nn.Module):
    def __init__(self):
        super(Mobilenet,self).__init__()

        def conv_bn(inp,oup,s):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3,s,1,bias= False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp,oup,s):
            return nn.Sequential(
                nn.Conv2d(inp,inp,3,s,1,groups=inp,bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp,oup,1,1,0,bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        self.model = nn.Sequential(
            conv_bn(3,32,2),
            conv_dw(32,32,1),

            conv_dw(32,64,2),
            conv_dw(64, 64, 1),

            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),

            conv_dw(128, 256, 2),
            conv_dw(256,256,1),
            nn.AvgPool2d(8)

        )
        self.fc = nn.Linear(256,152)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1,256)
        x = self.fc(x)
        return x



'''
model = Mobilenet().cuda()
img = torch.rand(4,3,64,64).cuda()   #416 320     800 608
out =  model(img)
print(out.size())
'''