import argparse
from dataloader.XJTU_loader import XJTUDdataset
from dataloader.MIT_loader import MITDdataset
from nets.Model import SOHMode
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser(description='A benchmark for SOH estimation')
    parser.add_argument('--random_seed', type=int, default=2023)
    # data
    parser.add_argument('--data', type=str, default='XJTU',choices=['XJTU','MIT'])
    parser.add_argument('--input_type',type=str,default='charge',choices=['charge','partial_charge','handcraft_features'])
    parser.add_argument('--test_battery_id',type=int,default=1,help='test battery id, 1-8 for XJTU (1-15 for batch-2), 1-5 for MIT')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--normalized_type',type=str,default='minmax',choices=['minmax','standard'])
    parser.add_argument('--minmax_range',type=tuple,default=(-1,1),choices=[(0,1),(-1,1)])
    parser.add_argument('--batch', type=int, default=1,choices=[1,2,3,4,5,6,7,8,9])

    # model
    parser.add_argument('--model',type=str,default='CNN',choices=['CNN','LSTM','GRU','MLP','Attention'])
    # CNN lr=2e-3  early_stop=30

    parser.add_argument('--lr',type=float,default=2e-3)
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--n_epoch',type=int,default=100)
    parser.add_argument('--early_stop',default=30)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--save_folder',default='results')
    parser.add_argument('--experiment_num',default=1,type=int,help='The number of times you want to repeat the same experiment')

    args = parser.parse_args()
    return args

def load_data(args,test_battery_id):
    if args.data == 'XJTU':
        loader = XJTUDdataset(args)
    else:
        loader = MITDdataset(args)

    if args.input_type == 'charge':
        data_loader = loader.get_charge_data(test_battery_id=test_battery_id)
    elif args.input_type == 'partial_charge':
        data_loader = loader.get_partial_data(test_battery_id=test_battery_id)
    else:
        data_loader = loader.get_features(test_battery_id=test_battery_id)
    return data_loader

def main(args):
    for batch in [6]:
        ids_list = [1,2,3,4,5,6,7,8]
        for test_id in ids_list:
            setattr(args,'batch',batch)
            for e in range(5):
                print()
                print(args.normalized_type,args.minmax_range,args.model,args.data, args.input_type,batch,test_id,e)
                try:
                    data_loader = load_data(args, test_battery_id=test_id)
                    model = SOHMode(args)
                    model.Train(data_loader['train'], data_loader['valid'], data_loader['test'],
                                save_folder=f'results/{args.data}-{args.input_type}/{args.model}/batch{batch}-testbattery{test_id}/experiment{e+1}',
                                )
                    del model
                    del data_loader
                    torch.cuda.empty_cache()

                except:
                    torch.cuda.empty_cache()
                    continue

def multi_task_XJTU():  # 一次性训练所有模型和所有输入类型
    args = get_args()
    setattr(args,'data','XJTU')
    for m in ['CNN','MLP','Attention','LSTM','GRU']:
        for type in ['handcraft_features','charge','partial_charge']:
            setattr(args, 'model', m)
            setattr(args, 'input_type',type)
            main(args)
            torch.cuda.empty_cache()

def multi_task_MIT():

    args = get_args()
    for norm in ['standard','minmax']:  # normalized_type
        setattr(args,'normalized_type',norm)
        setattr(args,'minmax_range',(0,1))
        setattr(args,'data','MIT')
        for m in ['GRU']:  # model
            for type in ['partial_charge']:
                setattr(args, 'model', m)
                setattr(args, 'input_type', type)
                for batch in range(1,10):   # batch
                    ids_list = [1, 2, 3, 4, 5]
                    for test_id in ids_list:   # test_battery_id
                        setattr(args, 'batch', batch)
                        for e in range(5):   # experiment
                            print()
                            print(args.model, args.data, args.input_type, batch, test_id, e)
                            try:
                                data_loader = load_data(args, test_battery_id=test_id)
                                model = SOHMode(args)
                                model.Train(data_loader['train'], data_loader['valid'], data_loader['test'],
                                            save_folder=f'results/{norm}/{args.data}-{args.input_type}/{args.model}/batch{batch}-testbattery{test_id}/experiment{e + 1}',
                                            )
                                del model
                                del data_loader
                                torch.cuda.empty_cache()
                            except:
                                torch.cuda.empty_cache()
                                continue

                torch.cuda.empty_cache()


if __name__ == '__main__':
    # if just want to train one model on one battery and one input type, use this:
    args = get_args()
    for e in range(args.experiment_num):
        data_loader = load_data(args, test_battery_id=args.test_battery_id)
        model = SOHMode(args)
        model.Train(data_loader['train'], data_loader['valid'], data_loader['test'],
                    save_folder=f'results/{args.data}-{args.input_type}/{args.model}/batch{args.batch}-testbattery{args.test_battery_id}/experiment{e + 1}',
                    )

    # multi_task_XJTU()
    # multi_task_MIT()

# 在程序中，我们提供了100个电池（我们自己实验生成了55个+MIT的45个）, with 3 input types and 3 normalization methods, 5 种基本模型供你选择
# 你可以自己选择训练哪些模型，哪些输入类型，哪些电池，哪些归一化方法，哪些超参数，哪些训练策略
# 你可以自己选择训练多少次，每次训练的结果都会保存在results文件夹中
# 我们这里只是提供一个baseline，你可以在此基础上进行改进，比如使用更好的模型，更好的超参数，更好的训练策略等等