import torch
import torch.nn as nn


class MLP(nn.Module):
    '''
    input shape: (N,4,128)
    '''

    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128*4,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,128),
            nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        '''
        :param x: (N,4,128)
        :return:
        '''
        x = x.view(-1,4*128)
        fea = self.net(x)
        out = self.predictor(fea)
        return out

if __name__ == '__main__':
    x = torch.rand(30,4,128)

    net = MLP()
    y = net(x)
    print(x.shape,y.shape)

    num_params = sum(param.numel() for param in net.parameters())
    print(num_params)