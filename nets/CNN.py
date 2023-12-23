import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),

            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out



class CNN(nn.Module):
    '''
    input shape: (N,4,128)
    '''
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = ResBlock(input_channel=4, output_channel=16, stride=1) # N,16,128
        self.layer2 = ResBlock(input_channel=16, output_channel=32, stride=2) # N,32,64
        self.layer3 = ResBlock(input_channel=32, output_channel=64, stride=2)  # N,64,32
        self.layer4 = ResBlock(input_channel=64, output_channel=96, stride=2)  # N,96,16
        self.layer5 = ResBlock(input_channel=96, output_channel=128, stride=2)  # N,128,8

        self.predictor = nn.Sequential(
            nn.Linear(128 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)

        )


    def forward(self,x):
        '''
        :param x: shape:(N,4,128)
        :return:
        '''
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        pred = self.predictor(out.view(out.size(0), -1))
        return pred

if __name__ == '__main__':
    x = torch.rand(30,4,128)

    net = CNN()
    y = net(x)
    print(x.shape,y.shape)

    num_params = sum(param.numel() for param in net.parameters())
    print(num_params)