import torch
import torch.nn as nn

class high_net(nn.Module):
    def __init__(self):
        super(high_net, self).__init__()
        self.conv41 = nn.Sequential(nn.Conv2d(11, 128, kernel_size=(4, 1),padding=(2,0)),nn.ReLU(True))
        self.conv14 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 4), padding=(0, 2)), nn.ReLU(True))
        self.conv33 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(3, 3),padding=(1,1)),nn.ReLU(True))
        self.conv22 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(2, 2)),nn.ReLU(True))

        self.conv00 = nn.Sequential(nn.Conv2d(11, 128, kernel_size=(2, 2),dilation = (3,3)),nn.ReLU(True))

        self.dense1 = nn.Sequential(nn.Linear(17 * 128 , 1024),nn.BatchNorm1d(1024),nn.ReLU(True))
        self.dense2 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(True))
        self.dense3 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True))

        self.output = nn.Linear(256, 4)

        self.initialize()

    def forward(self, x):
        x1 = self.conv41(x)
        x1 = self.conv14(x1)
        x1 = self.conv33(x1)
        x1 = self.conv22(x1)

        x2 = self.conv00(x)

        x = torch.cat((x1.view(-1, 16 * 128), x2.view(-1 , 128)),1)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.output(x)

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')

