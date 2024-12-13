# 改成从data_final里面调用包
import pandas as pd

from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred

from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack, Transformer
# 设置网格搜索和交叉验证
from sklearn.model_selection import GridSearchCV, KFold

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
# 固定numpy的随机种子
np.random.seed(10)

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.model_selection import GridSearchCV

class informer_gridsearch(Informer):
    def __init__(self, param_grid, cv=5, **kwargs):
        super().__init__(**kwargs)
        self.param_grid = param_grid
        self.cv = cv

    def fit(self, X, Y):
        # 将数据转换为PyTorch张量
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # 创建GridSearchCV对象
        grid_search = GridSearchCV(self, self.param_grid, cv=self.cv)

        # 使用GridSearchCV对象进行网格搜索和交叉验证
        grid_search.fit(X, Y)

        # 返回最佳的模型参数和效果
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        return best_params, best_score


class InformerWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, enc_in=13, dec_in=13, c_out=13, seq_len=24, label_len=12, out_len=1, d_model=512, n_heads=2, e_layers=2, dropout=0.05):
        self.model = Informer(enc_in=enc_in, dec_in=dec_in, c_out=c_out, seq_len=seq_len,
                              label_len=label_len, out_len=out_len, d_model=d_model, n_heads=n_heads, e_layers=e_layers, dropout=dropout)
        self.c_out = c_out
        self.enc_in = enc_in,
        self.dec_in = dec_in,
        self.seq_len = seq_len,
        self.label_len = label_len,
        self.factor = 10,
        self.d_model = d_model,
        self.n_heads = n_heads,
        self.e_layers = e_layers,
        # self.d_layers,
        # self.d_ff,
        self.dropout = dropout,
        self.out_len = out_len,
        # self.attn,
        # self.embed,
        # self.freq,
        # self.activation,
        # self.output_attention,
        # self.distil,
        # self.mix,
        # self.device

    def fit(self, X, y):
        self.model.train(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

class Exp_Informer(Exp_Basic):

    # 在类的定义中设置随机种子
    seed = 10
    torch.manual_seed(seed)  # 设置PyTorch的随机种子，从而控制CPU上的随机数生成器
    torch.cuda.manual_seed(seed)  # 设置GPU上的随机种子，以控制GPU上的随机数生成器
    torch.backends.cudnn.deterministic = True  # 可以使得PyTorch在使用CuDNN加速时，不再使用随机算法，而是使用确定性算法，从而保证每次运行结果一致
    torch.backends.cudnn.benchmark = False  # 可以关闭CuDNN的自动调整机制，从而保证每次运行时的性能稳定

    def __init__(self, args):

        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
            'Transformer': Transformer,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack' or self.args.model == 'Transformer':
            e_layers = self.args.e_layers if self.args.model == 'informer' or self.args.model == 'Transformer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 读入数据
    def _get_data(self, flag):
        args = self.args

        # 每个'Dataset_Custom'是一个类
        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,

            'FLX_AT-Neu_13': Dataset_Custom,
            'FLX_AU-DaS_13': Dataset_Custom,
            'FLX_AU-Dry_13': Dataset_Custom,
            'FLX_AU-Fog_13': Dataset_Custom,
            'FLX_AU-How_13': Dataset_Custom,
            'FLX_AU-Rig_13': Dataset_Custom,
            'FLX_AU-Stp_13': Dataset_Custom,
            'FLX_AU-Tum_13': Dataset_Custom,
            'FLX_AU-Wom_13': Dataset_Custom,

            'FLX_CA-Oas_13': Dataset_Custom,
            'FLX_CA-Qfo_13': Dataset_Custom,
            'FLX_CA-SF3_13': Dataset_Custom,
            'FLX_CA-TP1_13': Dataset_Custom,
            'FLX_CA-TP2_13': Dataset_Custom,
            'FLX_CA-TP3_13': Dataset_Custom,
            'FLX_CA-TP4_13': Dataset_Custom,
            'FLX_CH-Cha_13': Dataset_Custom,
            'FLX_CH-Fru_13': Dataset_Custom,

            'FLX_CH-Lae_13': Dataset_Custom,
            'FLX_CH-Oe1_13': Dataset_Custom,
            'FLX_CH-Oe2_13': Dataset_Custom,
            'FLX_CN-Cha_13': Dataset_Custom,
            'FLX_CN-Du2_13': Dataset_Custom,
            'FLX_CN-Ha2_13': Dataset_Custom,
            'FLX_CN-HaM_13': Dataset_Custom,

            'FLX_DE-Geb_13': Dataset_Custom,
            'FLX_DE-Hai_13': Dataset_Custom,
            'FLX_DE-Kli_13': Dataset_Custom,
            'FLX_DE-Lkb_13': Dataset_Custom,
            'FLX_DE-Lnf_13': Dataset_Custom,
            'FLX_DE-Tha_13': Dataset_Custom,
            'FLX_ES-LJu_13': Dataset_Custom,
            # 'FLX_FI-Hyy_13': {'data': 'FLX_FI-Hyy_13.csv', 'T': 'SWC_F_MDS_1', 'M': [13, 13, 13], 'S': [1, 1, 1], 'MS': [13, 13, 1]},
            'FLX_FI-Sod_13': Dataset_Custom,
            'FLX_GF-Guy_13': Dataset_Custom,
            'FLX_GH-Ank_13': Dataset_Custom,

            'FLX_IT-BCi_13': Dataset_Custom,
            'FLX_IT-CA1_13': Dataset_Custom,
            'FLX_IT-Cp2_13': Dataset_Custom,
            'FLX_IT-Noe_13': Dataset_Custom,
            'FLX_IT-Ren_13': Dataset_Custom,
            'FLX_IT-Ro1_13': Dataset_Custom,
            'FLX_IT-Ro2_13': Dataset_Custom,
            'FLX_IT-Tor_13': Dataset_Custom,
            'FLX_MY-PSO_13': Dataset_Custom,
            'FLX_PA-SPs_13': Dataset_Custom,
            'FLX_SD-Dem_13': Dataset_Custom,
            'FLX_US-AR1_13': Dataset_Custom,
            'FLX_US-ARM_13': Dataset_Custom,
            'FLX_US-Blo_13': Dataset_Custom,
            'FLX_US-CRT_13': Dataset_Custom,
            'FLX_US-IB2_13': Dataset_Custom,
            'FLX_US-KS2_13': Dataset_Custom,
            'FLX_US-Me2_13': Dataset_Custom,
            'FLX_US-Me3_13': Dataset_Custom,
            'FLX_US-Me5_13': Dataset_Custom,
            'FLX_US-Me6_13': Dataset_Custom,
            'FLX_US-MMS_13': Dataset_Custom,
            'FLX_US-Oho_13': Dataset_Custom,
            'FLX_US-SRC_13': Dataset_Custom,
            'FLX_US-SRG_13': Dataset_Custom,
            'FLX_US-SRM_13': Dataset_Custom,
            'FLX_US-Syv_13': Dataset_Custom,
            'FLX_US-Ton_13': Dataset_Custom,
            'FLX_US-UMB_13': Dataset_Custom,
            'FLX_US-UMd_13': Dataset_Custom,
            'FLX_US-Var_13': Dataset_Custom,
            'FLX_US-WCr_13': Dataset_Custom,
            'FLX_US-Whs_13': Dataset_Custom,
            'FLX_US-Wkg_13': Dataset_Custom,
            'FLX_ZM-Mon_13': Dataset_Custom,

            'FLX_AT-Neu': Dataset_Custom,
            'FLX_AU-DaS': Dataset_Custom,
            'FLX_AU-Dry': Dataset_Custom,
            'FLX_AU-Fog': Dataset_Custom,
            'FLX_AU-How': Dataset_Custom,
            'FLX_AU-Rig': Dataset_Custom,
            'FLX_AU-Stp': Dataset_Custom,
            'FLX_AU-Tum': Dataset_Custom,
            'FLX_AU-Wom': Dataset_Custom,

            'FLX_CA-Oas': Dataset_Custom,
            'FLX_CA-Qfo': Dataset_Custom,
            'FLX_CA-SF3': Dataset_Custom,
            'FLX_CA-TP1': Dataset_Custom,
            'FLX_CA-TP2': Dataset_Custom,
            'FLX_CA-TP3': Dataset_Custom,
            'FLX_CA-TP4': Dataset_Custom,
            'FLX_CH-Cha': Dataset_Custom,
            'FLX_CH-Fru': Dataset_Custom,
            'FLX_CH-Lae': Dataset_Custom,
            'FLX_CH-Oe1': Dataset_Custom,
            'FLX_CH-Oe2': Dataset_Custom,
            'FLX_CN-Cha': Dataset_Custom,
            'FLX_CN-Ha2': Dataset_Custom,
            'FLX_CN-Du2': Dataset_Custom,
            'FLX_CN-HaM': Dataset_Custom,

            'FLX_DE-Geb': Dataset_Custom,
            'FLX_DE-Hai': Dataset_Custom,
            'FLX_DE-Kli': Dataset_Custom,
            'FLX_DE-Lkb': Dataset_Custom,
            'FLX_DE-Lnf': Dataset_Custom,
            'FLX_DE-Tha': Dataset_Custom,
            'FLX_ES-LJu': Dataset_Custom,
            'FLX_FI-Sod': Dataset_Custom,
            'FLX_GF-Guy': Dataset_Custom,
            'FLX_GH-Ank': Dataset_Custom,

            'FLX_IT-BCi': Dataset_Custom,
            'FLX_IT-CA1': Dataset_Custom,
            'FLX_IT-Cp2': Dataset_Custom,
            'FLX_IT-Noe': Dataset_Custom,
            'FLX_IT-Ren': Dataset_Custom,
            'FLX_IT-Ro1': Dataset_Custom,
            'FLX_IT-Ro2': Dataset_Custom,
            'FLX_IT-Tor': Dataset_Custom,
            'FLX_MY-PSO': Dataset_Custom,
            'FLX_PA-SPs': Dataset_Custom,
            'FLX_SD-Dem': Dataset_Custom,
            'FLX_US-AR1': Dataset_Custom,
            'FLX_US-ARM': Dataset_Custom,
            'FLX_US-Blo': Dataset_Custom,
            'FLX_US-CRT': Dataset_Custom,
            'FLX_US-IB2': Dataset_Custom,
            'FLX_US-KS2': Dataset_Custom,
            'FLX_US-Me2': Dataset_Custom,
            'FLX_US-Me3': Dataset_Custom,
            'FLX_US-Me5': Dataset_Custom,
            'FLX_US-Me6': Dataset_Custom,
            'FLX_US-MMS': Dataset_Custom,
            'FLX_US-Oho': Dataset_Custom,
            'FLX_US-SRC': Dataset_Custom,
            'FLX_US-SRG': Dataset_Custom,
            'FLX_US-SRM': Dataset_Custom,
            'FLX_US-Syv': Dataset_Custom,
            'FLX_US-Ton': Dataset_Custom,
            'FLX_US-UMB': Dataset_Custom,
            'FLX_US-UMd': Dataset_Custom,
            'FLX_US-Var': Dataset_Custom,
            'FLX_US-WCr': Dataset_Custom,
            'FLX_US-Whs': Dataset_Custom,
            'FLX_US-Wkg': Dataset_Custom,
            'FLX_ZM-Mon': Dataset_Custom,

            'FLX_CN-Din_13': Dataset_Custom,
            'FLX_BE-Lon_13': Dataset_Custom,
            'FLX_CN-Qia_13': Dataset_Custom,
            'FLX_US-Goo_13': Dataset_Custom,
            'FLX_FR-LBr_13': Dataset_Custom,
            'FLX_NL-Loo_13': Dataset_Custom,
            'FLX_CA-TPD_13': Dataset_Custom,
            'FLX_FI-Hyy_13': Dataset_Custom,
            'FLX_BE-Vie_13': Dataset_Custom,
            'FLX_IT-Col_13': Dataset_Custom,

            'FLX_CN-Din': Dataset_Custom,
            'FLX_BE-Lon': Dataset_Custom,
            'FLX_CN-Qia': Dataset_Custom,
            'FLX_US-Goo': Dataset_Custom,
            'FLX_FR-LBr': Dataset_Custom,
            'FLX_NL-Loo': Dataset_Custom,
            'FLX_CA-TPD': Dataset_Custom,
            'FLX_FI-Hyy': Dataset_Custom,
            'FLX_BE-Vie': Dataset_Custom,
            'FLX_IT-Col': Dataset_Custom,

            'FLX_CN-Din_S': Dataset_Custom,
            'FLX_BE-Lon_S': Dataset_Custom,
            'FLX_CN-Qia_S': Dataset_Custom,
            'FLX_US-Goo_S': Dataset_Custom,
            'FLX_FR-LBr_S': Dataset_Custom,
            'FLX_NL-Loo_S': Dataset_Custom,
            'FLX_CA-TPD_S': Dataset_Custom,
            'FLX_FI-Hyy_S': Dataset_Custom,
            'FLX_BE-Vie_S': Dataset_Custom,
            'FLX_IT-Col_S': Dataset_Custom,

            'FLX_CN-Din_Causal': Dataset_Custom,
            'FLX_BE-Lon_Causal': Dataset_Custom,
            'FLX_CN-Qia_Causal': Dataset_Custom,
            'FLX_US-Goo_Causal': Dataset_Custom,
            'FLX_FR-LBr_Causal': Dataset_Custom,
            'FLX_NL-Loo_Causal': Dataset_Custom,
            'FLX_CA-TPD_Causal': Dataset_Custom,
            'FLX_FI-Hyy_Causal': Dataset_Custom,
            'FLX_BE-Vie_Causal': Dataset_Custom,
            'FLX_IT-Col_Causal': Dataset_Custom,

            'layer_l1_d1_SS': Dataset_Custom,
            'layer_l2_d1_SS': Dataset_Custom,
            'layer_l3_d1_SS': Dataset_Custom,
            'layer_l4_d1_SS': Dataset_Custom,
            'layer_l5_d1_SS': Dataset_Custom,
            'layer_l6_d1_SS': Dataset_Custom,
            'layer_l7_d1_SS': Dataset_Custom,
            'layer_l8_d1_SS': Dataset_Custom,
            'layer_l9_d1_SS': Dataset_Custom,
            'layer_l10_d1_SS': Dataset_Custom,
            'layer_l11_d1_SS': Dataset_Custom,
            'layer_l12_d1_SS': Dataset_Custom,
            'layer_l13_d1_SS': Dataset_Custom,
            'layer_l14_d1_SS': Dataset_Custom,
            'layer_l15_d1_SS': Dataset_Custom,

            'layer_l1_d1': Dataset_Custom,
            'layer_l2_d1': Dataset_Custom,
            'layer_l3_d1': Dataset_Custom,
            'layer_l4_d1': Dataset_Custom,
            'layer_l5_d1': Dataset_Custom,
            'layer_l6_d1': Dataset_Custom,
            'layer_l7_d1': Dataset_Custom,
            'layer_l8_d1': Dataset_Custom,
            'layer_l9_d1': Dataset_Custom,
            'layer_l10_d1': Dataset_Custom,

            'layer_l1_d2': Dataset_Custom,
            'layer_l2_d2': Dataset_Custom,
            'layer_l3_d2': Dataset_Custom,
            'layer_l4_d2': Dataset_Custom,
            'layer_l5_d2': Dataset_Custom,
            'layer_l6_d2': Dataset_Custom,
            'layer_l7_d2': Dataset_Custom,
            'layer_l8_d2': Dataset_Custom,
            'layer_l9_d2': Dataset_Custom,
            'layer_l10_d2': Dataset_Custom,

            'layer_l1_d3': Dataset_Custom,
            'layer_l2_d3': Dataset_Custom,
            'layer_l3_d3': Dataset_Custom,
            'layer_l4_d3': Dataset_Custom,
            'layer_l5_d3': Dataset_Custom,
            'layer_l6_d3': Dataset_Custom,
            'layer_l7_d3': Dataset_Custom,
            'layer_l8_d3': Dataset_Custom,
            'layer_l9_d3': Dataset_Custom,
            'layer_l10_d3': Dataset_Custom,

            'layer_l1_d4': Dataset_Custom,
            'layer_l2_d4': Dataset_Custom,
            'layer_l3_d4': Dataset_Custom,
            'layer_l4_d4': Dataset_Custom,
            'layer_l5_d4': Dataset_Custom,
            'layer_l6_d4': Dataset_Custom,
            'layer_l7_d4': Dataset_Custom,
            'layer_l8_d4': Dataset_Custom,
            'layer_l9_d4': Dataset_Custom,
            'layer_l10_d4': Dataset_Custom,

            'layer_l1_d5': Dataset_Custom,
            'layer_l2_d5': Dataset_Custom,
            'layer_l3_d5': Dataset_Custom,
            'layer_l4_d5': Dataset_Custom,
            'layer_l5_d5': Dataset_Custom,
            'layer_l6_d5': Dataset_Custom,
            'layer_l7_d5': Dataset_Custom,
            'layer_l8_d5': Dataset_Custom,
            'layer_l9_d5': Dataset_Custom,
            'layer_l10_d5': Dataset_Custom,

            'layer_l1_d6': Dataset_Custom,
            'layer_l2_d6': Dataset_Custom,
            'layer_l3_d6': Dataset_Custom,
            'layer_l4_d6': Dataset_Custom,
            'layer_l5_d6': Dataset_Custom,
            'layer_l6_d6': Dataset_Custom,
            'layer_l7_d6': Dataset_Custom,
            'layer_l8_d6': Dataset_Custom,
            'layer_l9_d6': Dataset_Custom,
            'layer_l10_d6': Dataset_Custom,

            'layer_l1_d7': Dataset_Custom,
            'layer_l2_d7': Dataset_Custom,
            'layer_l3_d7': Dataset_Custom,
            'layer_l4_d7': Dataset_Custom,
            'layer_l5_d7': Dataset_Custom,
            'layer_l6_d7': Dataset_Custom,
            'layer_l7_d7': Dataset_Custom,
            'layer_l8_d7': Dataset_Custom,
            'layer_l9_d7': Dataset_Custom,
            'layer_l10_d7': Dataset_Custom,

            'layer_l1_d8': Dataset_Custom,
            'layer_l2_d8': Dataset_Custom,
            'layer_l3_d8': Dataset_Custom,
            'layer_l4_d8': Dataset_Custom,
            'layer_l5_d8': Dataset_Custom,
            'layer_l6_d8': Dataset_Custom,
            'layer_l7_d8': Dataset_Custom,
            'layer_l8_d8': Dataset_Custom,
            'layer_l9_d8': Dataset_Custom,
            'layer_l10_d8': Dataset_Custom,

            'layer_l1_d9': Dataset_Custom,
            'layer_l2_d9': Dataset_Custom,
            'layer_l3_d9': Dataset_Custom,
            'layer_l4_d9': Dataset_Custom,
            'layer_l5_d9': Dataset_Custom,
            'layer_l6_d9': Dataset_Custom,
            'layer_l7_d9': Dataset_Custom,
            'layer_l8_d9': Dataset_Custom,
            'layer_l9_d9': Dataset_Custom,
            'layer_l10_d9': Dataset_Custom,

            'layer_l1_d10': Dataset_Custom,
            'layer_l2_d10': Dataset_Custom,
            'layer_l3_d10': Dataset_Custom,
            'layer_l4_d10': Dataset_Custom,
            'layer_l5_d10': Dataset_Custom,
            'layer_l6_d10': Dataset_Custom,
            'layer_l7_d10': Dataset_Custom,
            'layer_l8_d10': Dataset_Custom,
            'layer_l9_d10': Dataset_Custom,
            'layer_l10_d10': Dataset_Custom,

            'layer_l1_d11': Dataset_Custom,
            'layer_l2_d11': Dataset_Custom,
            'layer_l3_d11': Dataset_Custom,
            'layer_l4_d11': Dataset_Custom,
            'layer_l5_d11': Dataset_Custom,
            'layer_l6_d11': Dataset_Custom,
            'layer_l7_d11': Dataset_Custom,
            'layer_l8_d11': Dataset_Custom,
            'layer_l9_d11': Dataset_Custom,
            'layer_l10_d11': Dataset_Custom,

            'layer_l1_d12': Dataset_Custom,
            'layer_l2_d12': Dataset_Custom,
            'layer_l3_d12': Dataset_Custom,
            'layer_l4_d12': Dataset_Custom,
            'layer_l5_d12': Dataset_Custom,
            'layer_l6_d12': Dataset_Custom,
            'layer_l7_d12': Dataset_Custom,
            'layer_l8_d12': Dataset_Custom,
            'layer_l9_d12': Dataset_Custom,
            'layer_l10_d12': Dataset_Custom,

            'layer_l1_d13': Dataset_Custom,
            'layer_l2_d13': Dataset_Custom,
            'layer_l3_d13': Dataset_Custom,
            'layer_l4_d13': Dataset_Custom,
            'layer_l5_d13': Dataset_Custom,
            'layer_l6_d13': Dataset_Custom,
            'layer_l7_d13': Dataset_Custom,
            'layer_l8_d13': Dataset_Custom,
            'layer_l9_d13': Dataset_Custom,
            'layer_l10_d13': Dataset_Custom,

            'layer_l1_d14': Dataset_Custom,
            'layer_l2_d14': Dataset_Custom,
            'layer_l3_d14': Dataset_Custom,
            'layer_l4_d14': Dataset_Custom,
            'layer_l5_d14': Dataset_Custom,
            'layer_l6_d14': Dataset_Custom,
            'layer_l7_d14': Dataset_Custom,
            'layer_l8_d14': Dataset_Custom,
            'layer_l9_d14': Dataset_Custom,

            'layer_l1_d15': Dataset_Custom,
            'layer_l2_d15': Dataset_Custom,
            'layer_l3_d15': Dataset_Custom,
            'layer_l4_d15': Dataset_Custom,
            'layer_l5_d15': Dataset_Custom,
            'layer_l6_d15': Dataset_Custom,
            'layer_l7_d15': Dataset_Custom,
            'layer_l8_d15': Dataset_Custom,
            'layer_l9_d15': Dataset_Custom,
            'layer_l10_d15': Dataset_Custom,

        }
        # self = {Exp_Informer} <exp.exp_informer.Exp_Informer object at 0x000001908C251C30>
        Data = data_dict[self.args.data]
        # Data = {type} <class 'data.data_loader.Dataset_Custom'>
        # self.data = Data

        timeenc = 0 if args.embed != 'timeF' else 1  #  args.embed : time features encoding

        if flag == 'test':
            shuffle_flag = False;
            drop_last = False;      # 丢弃最后几个数据
            batch_size = args.batch_size;
            freq = args.freq;
            # inverse_t = True
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq;
            Data = Dataset_Pred
        else:   #  flag == 'train' or 'vali'
            shuffle_flag = True;  # True时需要将dropout设置得小一些，如0.05；False时需要设置大一些，如0.5
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq

        # dataset = Data() = data_dict[self.args.data]()
        # 从这里进入Dataset_custom
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,  #
            size=[args.seq_len, args.label_len, args.pred_len],  #  size:[24, 12, 1]
            features=args.features,  # MS
            target=args.target,      # SWC_F_MDS_1
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
            # scaler=args.scaler,
            # site_name = args.data
        )
        # features = data_set[1]
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            # scaler=args.scaler,
            # inverse=inverse_t
        )

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # 验证函数
    def vali(self, vali_data, vali_loader, criterion):

        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # 训练函数
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # train_loader  ???
        train_steps = len(train_loader)  #
        # len(self.data_x) - self.seq_len - self.pred_len + 1

        # 早停法
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()   # Adam
        criterion = self._select_criterion()   # MSELoss()

        #  amp : 全称为 Automatic mixed precision,自动混合精度
        #       可以在神经网络推理过程中，针对不同的层，采用不同的数据精度进行计算，从而实现节省显存和加快速度的目的
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            #
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                # if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                # iter_count = 0
                time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.requires_grad_(True)  # 这里的loss默认的requires_grad是False，因此在backward()处不会计算梯度，导致出错
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # 早停法
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    # 测试函数
    def test(self, setting, site_name, itr, num):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # # 对他们进行反归一化再计算评估指标
        preds = np.asarray(preds).reshape(preds.shape[0] * preds.shape[1], 1)
        preds = np.repeat(preds, num, axis=-1)
        trues = np.asarray(trues).reshape(trues.shape[0] * trues.shape[1], 1)
        trues = np.repeat(trues, num, axis=-1)
        # Dataset_Custom1= Dataset_Custom()
        # test_data就算是调用了那个函数
        # predstest = test_data.inverse_transform(preds)
        preds = test_data.inverse_transform(preds)
        trues = test_data.inverse_transform(trues)

        preds = preds[:, -1]
        trues = trues[:, -1]

        preds = preds.reshape(preds.shape[0], 1)
        trues = trues.reshape(trues.shape[0], 1)

        mae, mse, rmse, mape, mspe, r2, corr = metric(preds, trues)

        preds = preds.reshape(-1)
        trues = trues.reshape(-1)

        if itr == 1:
            plt.figure()
            plt.plot(preds, label="preds")
            plt.plot(trues, label="trues")
            plt.legend()

            # name = site_name + ' predict'
            # plt.title(name, size=11)
            # path = 'D:/05Result/01graph/informer/compare_all/' + site_name + '.png'
            # # path = 'D:/05Result/01graph/informer/compare_causal/' + site_name + '.png'
            # plt.savefig(path)

        # from sklearn.metrics import r2_score
        # print(r2_score(trues, preds))

        # df = pd.DataFrame(zip(preds, trues))
        df = pd.DataFrame({'pred': preds, 'true': trues})
        df.to_excel('D:/05Result/informer/72/compare_all/compare_all_' + site_name + '_48_24_12_0524.xlsx')
        # df.to_excel('D:/05Result/informer/72/compare_causal/compare_causal_' + site_name + '_0512.xlsx')

        print('rmse:{}, r2:{}, mae:{}, mape:{}, mse:{}, corr:{}, mspe:{}'.format(rmse, r2, mae, mape, mse, corr, mspe))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return rmse, r2, mae, mape, mse, corr, mspe

    # 预测函数
    def predict(self, setting, load=True):  # def predict(self, setting, load=False)

        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        # preds = np.array(preds)
        # ## 逆缩放
        # # preds = StandardScaler.inverse_transform(preds)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # preds = preds.reshape(-1)
        # trues = np.array(trues).reshape(-1)

        preds = np.array(preds)
        trues = np.array(trues)

        print('predict shape:', preds.shape, trues.shape)
        preds = np.array(preds).flatten()
        trues = np.array(trues).flatten()
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('predict shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe, r2, corr = metric(preds, trues)
        print('predict_metrics RMSE:', rmse, 'R2:', r2)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = outputs.detach().cpu().numpy()
            outputs = outputs.reshape(-1, 1)
            outputs = np.tile(outputs, 13)
            outputs = dataset_object.inverse_transform(outputs)
            outputs = outputs[:, -1]
            outputs = torch.tensor(outputs)
            outputs = outputs.unsqueeze(1)
            outputs = outputs.unsqueeze(2)
            outputs = outputs.cuda()

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
