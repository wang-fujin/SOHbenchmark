import torch
import torch.nn as nn
from nets.CNN import CNN
from nets.LSTM import LSTM
from nets.Attention import Attention
from nets.GRU import GRU
from nets.MLP import MLP
from utils.util import AverageMeter,eval_metrix
import numpy as np
import matplotlib.pyplot as plt
import os

class SOHMode(nn.Module):
    '''
    data shape:
    charge_data (N,4,128)
    partial_data (N,4,128)
    features (N,1,67)
    '''
    def __init__(self,args):
        super(SOHMode,self).__init__()
        self.args = args
        self.pre_net = self._preprocessing_net()
        self.backbone = self._backbone()
        self.pre_net.to(args.device)
        self.backbone.to(args.device)
        self._initialize_weights()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                [30,70],
                gamma=0.5,
            )

        self.mse = torch.nn.MSELoss()
        self.loss_meter = AverageMeter()
        self.best_state = None




    def _preprocessing_net(self):
        '''
        A preprocess network which transform data from different sources into the same shape
        :return: A network, with output shape (N,4,128)
        '''
        if self.args.input_type == 'charge': # charge_data (N,4,128)
            net = nn.Conv1d(in_channels=4,out_channels=4,kernel_size=1)
        elif self.args.input_type == 'partial_charge': #partial_data (N,4,128)
            net = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)
        else:  # features (N,1,67)
            net = nn.Linear(67,128*4) #(N,128*4)

        return net


    def _backbone(self):
        backbone = eval(self.args.model)()
        return backbone

    def forward(self,x):
        if self.args.input_type == 'handcraft_features':
            x = self.pre_net(x)
            x = x.view(-1,4,128)
        out = self.backbone(x)
        return out

    def _train_one_epoch(self,train_loader):
        self.pre_net.train()
        self.backbone.train()
        self.loss_meter.reset()
        for data,label in train_loader:
            data,label = data.to(self.args.device),label.to(self.args.device)
            pred = self.forward(data)
            loss = self.mse(pred, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_meter.update(loss.item())

    def predict(self,test_loader):
        self.pre_net.eval()
        self.backbone.eval()
        self.loss_meter.reset()
        with torch.no_grad():
            true_label = []
            pred_label = []
            for data, label in test_loader:
                data, label = data.to(self.args.device), label.to(self.args.device)
                pred = self.forward(data)
                loss = self.mse(pred, label)

                self.loss_meter.update(loss.item())
                true_label.append(label.cpu().detach().numpy())
                pred_label.append(pred.cpu().detach().numpy())
            true_label = np.concatenate(true_label)
            pred_label = np.concatenate(pred_label)
        return true_label,pred_label

    def Train(self,train_loader,valid_loader,test_loader,save_folder=None):
        min_loss = 10
        stop = 0
        self.train_loss = []
        self.valid_loss = []
        self.true_label, self.pred_label = None, None


        for e in range(1,self.args.n_epoch+1):
            self._train_one_epoch(train_loader)
            self.scheduler.step()
            train_l = self.loss_meter.avg
            self.train_loss.append(train_l)
            stop += 1

            self.predict(valid_loader)
            valid_l = self.loss_meter.avg
            self.valid_loss.append(valid_l)

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(
                f"\r epoch=[{e}/{self.args.n_epoch}]  train loss : {train_l:.5f}  valid loss : {valid_l:.5f}  lr : {lr:.5f} ",
                end=''
            )
            if e % 10 == 0:
                print("")

            if valid_l < min_loss:
                self.best_state = {'pre_net': self.pre_net.state_dict(),
                                   'backbone':self.backbone.state_dict()}
                self.true_label, self.pred_label = self.predict(test_loader)
                print(f' ------ test loss : {self.loss_meter.avg:.5f}')
                min_loss = valid_l
                stop = 0
            if stop >= self.args.early_stop:
                break
        if save_folder is not None:
            self.save_all(save_folder)


    def save_all(self,folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        prefix = self.args.model + '_' + self.args.input_type
        errors = eval_metrix(self.true_label,self.pred_label)
        np.savez(os.path.join(folder,f'{prefix}_results.npz'),
                 train_loss = np.array(self.train_loss),
                 valid_loss = np.array(self.valid_loss),
                 true_label = np.array(self.true_label),
                 pred_label = np.array(self.pred_label),
                 test_errors = np.array(errors)
                 )
        torch.save(self.best_state, os.path.join(folder,f'{prefix}_model.pkl'))


    def _plot_loss(self,train_loss,valid_loss):

        self.fig_loss = plt.figure()
        plt.plot(train_loss,label='train')
        plt.plot(valid_loss,label='valid')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()
        plt.close()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)








if __name__ == '__main__':
    def get_args():
        import argparse
        parser = argparse.ArgumentParser(description='A benchmark for SOH estimation')
        parser.add_argument('--random_seed', type=int, default=2023)
        # data
        parser.add_argument('--data', type=str, default='XJTU', choices=['XJTU', 'MIT'])
        parser.add_argument('--input_type', type=str, default='charge',
                            choices=['charge', 'partial_charge', 'handcraft_features'])
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--normalized_type', type=str, default='minmax', choices=['minmax', 'standard'])
        parser.add_argument('--minmax_range', type=tuple, default=(0, 1), choices=[(0, 1), (1, 1)])
        parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9])

        # model
        parser.add_argument('--model', type=str, default='LSTM', choices=['CNN', 'LSTM', 'GRU', 'MLP', 'Attention'])

        parser.add_argument('--lr', type=float, default=2e-3)
        parser.add_argument('--weight_decay', default=5e-4)
        parser.add_argument('--n_epoch', type=int, default=500)
        parser.add_argument('--early_stop', default=20)
        parser.add_argument('--device', default='cuda')

        args = parser.parse_args()
        return args


    args = get_args()
    model = SOHMode(args)

    x1 = torch.rand(30,4,128).to('cuda')
    x2 = torch.rand(30,1,67)

    y = model(x1)
    print(model)
    print('output shape:',y.shape)