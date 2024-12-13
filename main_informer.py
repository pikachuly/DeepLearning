import argparse
import os
import torch
import pandas as pd
from exp.exp_informer import Exp_Informer
import csv
from sklearn.model_selection import GridSearchCV

use_sites_77 = ['AT-Neu','AU-DaS', 'AU-Dry', 'AU-Fog',
             'AU-How', 'AU-Rig', 'AU-Stp', 'AU-Tum',
             'AU-Wom', 'BE-Lon', 'BE-Vie', 'CA-Oas',
             'CA-Qfo', 'CA-SF3', 'CA-TP1', 'CA-TP2',
             'CA-TP3', 'CA-TP4', 'CA-TPD', 'CH-Cha',
             'CH-Fru', 'CH-Lae', 'CH-Oe1', 'CH-Oe2',
             'CN-Cha', 'CN-Din', 'CN-Du2', 'CN-Ha2',
             'CN-HaM', 'CN-Qia', 'DE-Geb', 'DE-Hai',
             'DE-Kli', 'DE-Lkb', 'DE-Lnf', 'DE-Tha',
             'ES-LJu', 'FI-Hyy', 'FI-Sod', 'GF-Guy',
             'GH-Ank', 'IT-BCi', 'IT-CA1', 'IT-Cp2',
             'IT-Noe', 'IT-Ren', 'IT-Ro1', 'IT-Ro2',
             'IT-Tor', 'MY-PSO', 'PA-SPs', 'SD-Dem',
             'US-AR1', 'US-ARM', 'US-Blo', 'US-CRT',
             'US-Goo', 'US-IB2', 'US-KS2', 'US-Me2',
             'US-Me3', 'US-Me5', 'US-Me6', 'US-MMS',
             'US-Oho', 'US-SRC', 'US-SRG', 'US-SRM',
             'US-Syv', 'US-Ton', 'US-UMB', 'US-UMd',
             'US-Var', 'US-WCr', 'US-Whs','US-Wkg',
             'ZM-Mon']

useless_sites = ['CA-Oas', 'CH-Lae', 'CH-Oe1', 'CH-Oe2',
                 'CN-Cha', 'CN-Din', 'CN-Ha2', 'CN-Qia',
                 'GF-Guy', 'IT-Ro1', 'MY-PSO', 'PA-SPs',
                 'US-Goo', 'US-Me3', 'US-UMB', 'US-UMd']

use_sites_61 = [
             'AT-Neu', 'AU-DaS', 'AU-Dry', 'AU-Fog',
             'AU-How', 'AU-Rig', 'AU-Stp', 'AU-Tum',
             'AU-Wom', 'BE-Lon',
             'BE-Vie', 'CA-Qfo',
             'CA-SF3', 'CA-TP1', 'CA-TP2', 'CA-TP3',
             'CA-TP4', 'CA-TPD', 'CH-Cha',
             'CH-Fru',
             'CN-Du2', 'CN-HaM', 'DE-Geb', 'DE-Hai',
             'DE-Kli', 'DE-Lkb', 'DE-Lnf', 'DE-Tha',
             'ES-LJu', 'FI-Hyy', 'FI-Sod', 'GH-Ank',
             'IT-BCi', 'IT-CA1', 'IT-Cp2', 'IT-Noe',
             'IT-Ren', 'IT-Ro2', 'IT-Tor', 'SD-Dem',
             'US-AR1', 'US-ARM', 'US-Blo', 'US-CRT',
             'US-IB2', 'US-KS2', 'US-Me2',
             'US-Me5',
             'US-Me6', 'US-MMS', 'US-Oho', 'US-SRC',
             'US-SRG', 'US-SRM', 'US-Syv', 'US-Ton',
             'US-Var', 'US-WCr', 'US-Whs', 'US-Wkg',
             'ZM-Mon']

use_sites_72 = ['AT-Neu','AU-DaS', 'AU-Dry', 'AU-Fog',
             'AU-How', 'AU-Rig', 'AU-Stp', 'AU-Tum',
             'AU-Wom', 'BE-Lon', 'BE-Vie', 'CA-Oas',
             'CA-Qfo', 'CA-SF3', 'CA-TP1', 'CA-TP2',
             'CA-TP3', 'CA-TP4', 'CA-TPD', 'CH-Cha',
             'CH-Fru',  'CH-Oe1', 'CH-Oe2',
             'CN-Cha', 'CN-Din', 'CN-Du2', 'CN-Ha2',
             'CN-HaM', 'CN-Qia', 'DE-Geb', 'DE-Hai',
             'DE-Kli', 'DE-Lkb', 'DE-Lnf', 'DE-Tha',
             'ES-LJu', 'FI-Hyy', 'FI-Sod', 'GF-Guy',
             'GH-Ank', 'IT-BCi', 'IT-CA1',
             'IT-Noe', 'IT-Ren', 'IT-Ro1', 'IT-Ro2',
             'IT-Tor', 'MY-PSO', 'PA-SPs', 'SD-Dem',
             'US-AR1', 'US-ARM', 'US-Blo',
             'US-Goo', 'US-IB2', 'US-KS2', 'US-Me2',
             'US-Me3', 'US-Me5', 'US-Me6', 'US-MMS',
             'US-Oho', 'US-SRC', 'US-SRG',
             'US-Syv', 'US-Ton', 'US-UMB', 'US-UMd',
             'US-Var', 'US-WCr', 'US-Whs', 'US-Wkg']

delete = ['ZM-Mon', 'CH-Lae', 'IT-Cp2', 'US-CRT', 'US-SRM']

SITE_NAME = []
RMSE = []
R2 = []
MAE = []
MAPE = []
MSE = []
CORR = []
MSPE = []
results = []

SITE_NAME_2 = []
RMSE_2 = []
R2_2 = []
MAE_2 = []
MAPE_2 = []
MSE_2 = []
CORR_2 = []
MSPE_2 = []
results_2 = []

# import numpy as np
# # 固定numpy的随机种子
# np.random.seed(10)
# # 在类的定义中设置随机种子
# seed = 10
# torch.manual_seed(seed)  # 设置PyTorch的随机种子，从而控制CPU上的随机数生成器
# torch.cuda.manual_seed(seed)  # 设置GPU上的随机种子，以控制GPU上的随机数生成器
# torch.backends.cudnn.deterministic = True  # 可以使得PyTorch在使用CuDNN加速时，不再使用随机算法，而是使用确定性算法，从而保证每次运行结果一致
# torch.backends.cudnn.benchmark = False  # 可以关闭CuDNN的自动调整机制，从而保证每次运行时的性能稳定

downsites = ['AU-How', 'CA-TP1', 'CA-TP3', 'CA-TP4', 'CH-Lae', 'IT-CA1', 'IT-Noe', 'IT-Ro2',
             'US-Blo', 'US-CRT', 'US-Goo', 'US-IB2', 'US-Me2', 'US-Me5', 'US-Oho', 'US-SRM',
             'US-WCr', 'US-Whs', 'US-Wkg']


# c = [0.7, 0.8, 1.0]
n_seq = [12,24,36,48,72,96]
for seq in n_seq:
    # seq = 3
    # for i in useless_sites:
    # for i in use_sites_77:
    for i in use_sites_72:
        # i = 'US-KS2'
        causal = 0.8    # 前八份因果强度特征选择结果
        # causal = 0    # 设置causal=0时检查数据集
        # rf = 0.1    # 前八份rf特征选择结果
        # corr = 1   # 前八份corr特征选择结果

        print('='*150)
        # print('=================================================== causal = ', causal, '======================================================')
        parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
        # 使用的网络结构（方便对比实验），使用defalut更改网络结构
        parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD), Transformer]')
        # 修改数据部分

        # site_name = 'FLX_' + i + '_13_del'
        site_name = 'FLX_' + i + '_13'
        # site_name = 'FLX_' + i

        # 读的数据是什么（类型 路径）
        parser.add_argument('--data', type=str, default=site_name, help='data')
        # all     path = 'D:/FluxNet/DD/Causal/all77_inter3/FLX_' + site_name + '_13.csv'
        # causal  path = 'D:/FluxNet/DD/Causal/all77_inter3/FLX_' + site_name + '.csv'
        # parser.add_argument('--root_path', type=str, default='D:/FluxNet/DD/Causal/77/all77_delete', help='root path of the data file')
        # parser.add_argument('--root_path', type=str, default='D:/FluxNet/DD/Causal/77/all77_abnormal/features_select', help='root path of the data file')
        # parser.add_argument('--root_path', type=str, default='D:/FluxNet/DD/Causal/77/causal77_line_' + str(causal), help='root path of the data file')
        parser.add_argument('--root_path', type=str, default='D:/FluxNet/DD/Causal/77/all77_line', help='root path of the data file')

        # parser.add_argument('--root_path', type=str, default='D:/FluxNet/DD/Causal/77/all77_bad', help='root path of the data file')
        # parser.add_argument('--root_path', type=str, default='D:/FluxNet/DD/Causal/77/causal77_line', help='root path of the data file')
        # parser.add_argument('--root_path', type=str, default='data/EachLayer/DD_77/causal', help='root path of the data file')

        # parser.add_argument('--root_path', type=str, default='D:/FluxNet/DD/RF/rf77_line_' + str(rf), help='root path of the data file')

        parser.add_argument('--data_path', type=str, default=site_name + '.csv', help='data file')
        # 预测的种类及方法
        parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        # 哪一列要当作是标签
        parser.add_argument('--target', type=str, default='SWC_F_MDS_1', help='target feature in S or MS task')
        # 数据中存在时间 时间是以什么为单位（属于数据挖掘中的重采样）
        parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        # 模型最后保存位置
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
        '''
        96 48 24 -> 48 24 12
        5.066084384918213 -> 4.0803375244140625
        48 24 12 -> 32 18 96
        4.0803375244140625 -> 3.204860210418701
        32 18 9 -> 24 12 6
        3.204860210418701 -> 2.4911155700683594
        24 12 6 -> 12 6 3
        2.4911155700683594 -> 1.80740225315094
        12 6 3  -> 6 3 1
        1.80740225315094 -> 1.2889044284820557
        6 3 1 -> 4 2 1
        1.2889044284820557 -> 1.1285794973373413
        '''
        # 当前输入序列长度（可自定义）
        parser.add_argument('--seq_len', type=int, default=24, help='input sequence length of Informer encoder')
        # 标签（带着预测值的那个东西）长度（可自定义）
        parser.add_argument('--label_len', type=int, default=12, help='start token length of Informer decoder')
        # 预测未来序列长度（可自定义）
        parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
        # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

        # LSTM 看看结果是否相差很大
        # transformer 原理 实用性
        # 比较结果 结合优势
        # 数据集：墒情预测数据
        # 参数校准 深度学习 机器学习做后处理
        # 用经验建立起来的模型

        # 编码器、解码器输入输出维度
        # default 7->17
        parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
        # 输出预测未来多少个值
        parser.add_argument('--c_out', type=int, default=8, help='output size')

        # 隐层特征
        # 512->256
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        # 多头注意力机制
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        # 要做几次多头注意力机制
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        # 堆叠几层encoder
        parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        # 对Q进行采样，对Q采样的因子数
        parser.add_argument('--factor', type=int, default=10, help='probsparse attn factor') # 5 -> 10
        parser.add_argument('--padding', type=int, default=0, help='padding type')  #
        # 是否下采样操作pooling
        parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)

        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')  # 0.05
        # 注意力机制
        parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
        parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, ons:[timeF, fixed, learned]')

        parser.add_argument('--activation', type=str, default='gelu',help='activation')

        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data', default=True)
        parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)

        # 读数据
        parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
        # windows用户只能给0
        parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        # 训练轮数
        parser.add_argument('--itr', type=int, default=2, help='experiments times')  # 实验次数为两次
        # 增加训练轮数 # 训练迭代次数
        parser.add_argument('--train_epochs', type=int, default=32, help='train epochs')
        # 增加了batch_size 损失函数降低了
        # 32：mse:1.2867071628570557, mae:0.8785622715950012
        # 64：mse:1.6609605550765991, mae:1.0216920375823975
        # minibatch 大小
        parser.add_argument('--batch_size', type=int, default=20, help='batch size of train input data')  # 32->10
        # 早停策略
        # 3 => 7 最佳
        # 超过n次loss没有下降就停止了
        # 如果patience=0,则不进行early stopping
        # --------------------------------------------------------------------------------------------------------------
        parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
        # 学习率 0.0001为最佳
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test',help='exp description')
        # 损失函数
        # mse -> rmse
        parser.add_argument('--loss', type=str, default='mse', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        # 是否为分布式
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
        ######################################################## 0505
        # 是否反归一化输出值
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        # 如果为分布式指定有几个显卡
        parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
        parser.add_argument('--scaler', type=bool, default=True, help='inverse_transform')

        args = parser.parse_args()

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ','')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]


        # path = 'D:/FluxNet/DD/RF/rf77_line_0.1/FLX_' + i + '.csv'
        # path = 'D:/FluxNet/DD/Causal/77/all77_delete/FLX_' + i + '_13.csv'
        path = 'D:/FluxNet/DD/Causal/77/all77_line/FLX_' + i + '_13.csv'
        # path = 'D:/FluxNet/DD/Causal/77/causal77_line_' + str(causal) + '/FLX_' + i + '.csv'
        data = pd.read_csv(path, header=0, index_col=None)
        num = len(data.columns) - 1
        data_parser = {
            # 'FLX_' + i + '_13_del': {'data': 'FLX_' + i + '_13_del.csv', 'T': 'SWC_F_MDS_1', 'M': [num, num, num], 'S': [1, 1, 1], 'MS': [num, num, 1]},
            'FLX_' + i + '_13': {'data': 'FLX_' + i + '_13.csv', 'T': 'SWC_F_MDS_1', 'M': [num, num, num], 'S': [1, 1, 1], 'MS': [num, num, 1]},
            # 'FLX_' + i: {'data': 'FLX_' + i + '.csv', 'T': 'SWC_F_MDS_1', 'M': [num, num, num], 'S': [1, 1, 1], 'MS': [num, num, 1]},
        }

        # if rf == 0.1:
        #     data_parser = rf_val.read(i, num=len(data.columns) - 1)

        if args.data in data_parser.keys():
            data_info = data_parser[args.data]
            args.data_path = data_info['data']
            args.target = data_info['T']
            args.enc_in, args.dec_in, args.c_out = data_info[args.features]

        args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ',' ').split(',')]
        args.detail_freq = args.freq
        args.freq = args.freq[-1:]

        print('Args in experiment:')
        print(args)

        Exp = Exp_Informer

        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                        args.seq_len, args.label_len, args.pred_len,
                        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
                        args.embed, args.distil, args.mix, args.des, ii)

            exp = Exp(args) # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            rmse, r2, mae, mape, mse, corr, mspe = exp.test(setting, site_name, ii, num)

            if ii == 0:
                SITE_NAME.append(i)
                RMSE.append(rmse)
                R2.append(r2)
                MAE.append(mae)
                MSE.append(mse)
                CORR.append(corr)
                MAPE.append(mape)
                MSPE.append(mspe)
                # path = r'D:\Result\informer\check\check_causal77_line_patience=7_itr1_0410.csv'
                # with open(path, mode='a', newline="") as file:
                #     writer = csv.writer(file)
                #     writer.writerow([i, rmse, r2, mae, mape])

            if ii == 1:
                SITE_NAME_2.append(i)
                RMSE_2.append(rmse)
                R2_2.append(r2)
                MAE_2.append(mae)
                MSE_2.append(mse)
                CORR_2.append(corr)
                MAPE_2.append(mape)
                MSPE_2.append(mspe)

            results = pd.DataFrame({'SITE_NAME': SITE_NAME, 'RMSE': RMSE, 'R2': R2, 'MAE': MAE, 'CORR': CORR, 'MSE': MSE, 'MAPE': MAPE, 'MSPE': MSPE})
            results_2 = pd.DataFrame({'SITE_NAME_2': SITE_NAME_2, 'RMSE_2': RMSE_2, 'R2_2': R2_2, 'MAE_2': MAE_2, 'CORR_2': CORR_2, 'MSE_2': MSE_2, 'MAPE_2': MAPE_2,  'MSPE_2': MSPE_2})
            torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
        # for i in use_sites_77:
        break
    # path = r'D:\05Result\informer\Supplementary experiments\itr1\all72_informer_itr1_patience=7_24_12_'+str(seq)+'_0706.xlsx'
    # results.to_excel(path)
    # path = r'D:\05Result\informer\Supplementary experiments\itr2\all72_informer_itr2_patience=7_24_12_'+str(seq)+'_0706.xlsx'
    # results_2.to_excel(path)
    # causal = 0.7 0.8
    break


    # seq = [1,3,7,12,14,21,24]
    # break
    # else:
    #     continue
        # if args.do_predict:
        # print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.predict(setting, True)


