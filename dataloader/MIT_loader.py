from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from utils.Scaler import Scaler

class MITDdataset():
    def __init__(self,args):
        super(MITDdataset).__init__()
        self.root = 'data/MIT'
        self.max_capacity = 1.1
        self.normalized_type = args.normalized_type
        self.minmax_range = args.minmax_range
        self.seed = args.random_seed
        self.batch = args.batch
        self.batch_size = args.batch_size
        if self.batch > 9:
            raise IndexError(f'"batch" must be in the [1, 9], but got {self.batch}. ')


    def _parser_mat_data(self,battery_i_mat):
        '''
        :param battery_i_mat: shape:(1,len)
        :return: np.array
        '''
        data = []
        label = []
        for i in range(battery_i_mat.shape[1]):
            cycle_i_data = battery_i_mat[0,i]
            time = cycle_i_data['relative_time_min'] # (1,128)
            current = cycle_i_data['current_A'] # (1,128)
            voltage = cycle_i_data['voltage_V'] # (1,128)
            temperature = cycle_i_data['temperature_C'] # (1,128)
            capacity = cycle_i_data['capacity'][0]
            label.append(capacity)
            cycle_i = np.concatenate([time,current,voltage,temperature],axis=0)
            data.append(cycle_i)
        data = np.array(data,dtype=np.float32)
        label = np.array(label,dtype=np.float32)
        print(data.shape,label.shape)

        scaler = Scaler(data)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)
        soh = label / self.max_capacity

        return data,soh

    def _encapsulation(self,train_x,train_y,test_x,test_y):
        '''
        Encapsulate the numpy.array into DataLoader
        :param train_x: numpy.array
        :param train_y: numpy.array
        :param test_x: numpy.array
        :param test_y: numpy.array
        :return:
        '''
        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        test_x = torch.from_numpy(test_x)
        test_y = torch.from_numpy(test_y)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=self.seed)
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True,
                                  drop_last=False)
        valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=self.batch_size, shuffle=True,
                                  drop_last=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader

    def _get_raw_data(self,path,test_battery_id):
        mat = loadmat(path)
        battery = mat['battery']
        battery_ids = list(range(1, battery.shape[1] + 1))
        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in the {battery_ids}, but got {test_battery_id}. ')

        test_battery = battery[0, test_battery_id - 1][0]
        print(f'test battery id: {test_battery_id}, test data shape: ', end='')
        test_x, test_y = self._parser_mat_data(test_battery)
        train_x, train_y = [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            print(f'train battery id: {id}, ', end='')
            train_battery = battery[0, id - 1][0]
            x, y = self._parser_mat_data(train_battery)
            train_x.append(x)
            train_y.append(y)
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        print('train data shape: ', train_x.shape, train_y.shape)

        return self._encapsulation(train_x, train_y, test_x, test_y)

    def get_charge_data(self,test_battery_id=1):
        print('----------- load charge data -------------')
        charge_files = os.listdir(os.path.join(self.root, 'charge'))
        file_name = charge_files[self.batch-1]

        self.charge_path = os.path.join(self.root, 'charge', file_name)
        train_loader, valid_loader, test_loader = self._get_raw_data(path=self.charge_path,test_battery_id=test_battery_id)
        data_dict = {'train':train_loader,
                     'test':test_loader,
                     'valid':valid_loader}
        print('-------------  finished !  ---------------')
        return data_dict


    def get_partial_data(self,test_battery_id=1):
        print('----------- load partial_charge data -------------')
        charge_files = os.listdir(os.path.join(self.root, 'partial_charge'))
        file_name = charge_files[self.batch - 1]
        self.partial_path = os.path.join(self.root, 'partial_charge', file_name)
        train_loader, valid_loader, test_loader = self._get_raw_data(path=self.partial_path,
                                                                     test_battery_id=test_battery_id)
        data_dict = {'train': train_loader,
                     'test': test_loader,
                     'valid': valid_loader}
        print('----------------  finished !  --------------------')
        return data_dict

    def _parser_xlsx(self,df_i):
        '''
        features dataframe
        :param df_i: shape:(N,C+1)
        :return:
        '''
        N = df_i.shape[0]
        x = np.array(df_i.iloc[:, :-1],dtype=np.float32)
        label = np.array(df_i['label'],dtype=np.float32).reshape(-1, 1)

        scaler = Scaler(x)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)
        soh = label / self.max_capacity

        return data, soh

    def get_features(self,test_battery_id=1):
        print('----------- load features -------------')
        charge_files = os.listdir(os.path.join(self.root, 'handcraft_features'))
        file_name = charge_files[self.batch - 1]
        self.features_path = os.path.join(self.root, 'handcraft_features', file_name)
        df = pd.read_excel(self.features_path,sheet_name=None)
        sheet_names = list(df.keys())
        battery_ids = list(range(1, len(sheet_names)+1))

        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in the {battery_ids}, but got {test_battery_id}. ')
        test_battery_df = pd.read_excel(self.features_path,sheet_name=test_battery_id-1,header=0)
        test_x,test_y = self._parser_xlsx(test_battery_df)
        print(f'test battery id: {test_battery_id}, test data shape: {test_x.shape}, {test_y.shape}')

        train_x, train_y = [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            sheet_name = sheet_names[id-1]
            df_i = df[sheet_name]
            x, y = self._parser_xlsx(df_i)
            print(f'train battery id: {id}, {x.shape}, {y.shape}')
            train_x.append(x)
            train_y.append(y)
        train_x = np.concatenate(train_x,axis=0)
        train_y = np.concatenate(train_y,axis=0)
        print('train data shape: ', train_x.shape, train_y.shape)

        train_loader, valid_loader, test_loader = self._encapsulation(train_x, train_y, test_x, test_y)
        data_dict = {'train': train_loader,
                     'test': test_loader,
                     'valid': valid_loader}
        print('---------------  finished !  ----------------')
        return data_dict





if __name__ == '__main__':
    import argparse
    def get_args():

        parser = argparse.ArgumentParser(description='dataloader test')
        parser.add_argument('--random_seed',type=int,default=2023)
        # data
        parser.add_argument('--data', type=str, default='MIT', choices=['XJTU', 'MIT'])
        parser.add_argument('--input_type', type=str, default='charge',
                            choices=['charge', 'partial_charge', 'handcraft_features'])
        parser.add_argument('--normalized_type', type=str, default='minmax', choices=['minmax', 'standard'])
        parser.add_argument('--minmax_range', type=tuple, default=(0, 1), choices=[(0, 1), (1, 1)])
        parser.add_argument('--batch_size', type=int, default=128)
        # the parameters for XJTU data
        parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5])

        args = parser.parse_args()
        return args

    args = get_args()
    data = MITDdataset(args)
    charge_data = data.get_charge_data(test_battery_id=3)
    partial_charge = data.get_partial_data(test_battery_id=5)
    features = data.get_features(test_battery_id=2)
