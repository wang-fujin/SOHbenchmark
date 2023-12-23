import torch
import torch.nn as nn

class Attention(nn.Module):
    '''
    input shape: (N,4,128)
    '''

    def __init__(self):
        super(Attention, self).__init__()
        self.dmodel = 128
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dmodel,nhead=4,dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=2)
        self.linear = nn.Linear(4,self.dmodel)
        self.conv = nn.Conv1d(in_channels=self.dmodel,out_channels=8,kernel_size=1)
        self.predictor = nn.Sequential(
            nn.Linear(128*8, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        '''
        :param x: (N,4,128)  (N,C,L)
        :return:
        '''
        x = x.transpose(1, 2)  # (N,L,C)
        x = self.linear(x)
        out = self.transformer_encoder(x)  #(N,L,d_model)
        fea = self.conv(out.transpose(1, 2)) #(N,d_model,L) -> (N,8,L)
        pred = self.predictor(fea.view(fea.shape[0],-1))
        return pred


if __name__ == '__main__':
    x = torch.rand((10, 4, 128)) #(N,C,L)

    net = Attention()
    y = net(x)
    print(x.shape,y.shape)

    num_params = sum(param.numel() for param in net.parameters())
    print(num_params)